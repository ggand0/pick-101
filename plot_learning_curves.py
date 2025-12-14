"""Plot learning curves from tensorboard logs."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(log_dir: Path) -> dict[str, list]:
    """Load scalar data from tensorboard event files."""
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={event_accumulator.SCALARS: 0},  # Load all scalars
    )
    ea.Reload()

    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }
    return data


def smooth(values: list, weight: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_comparison(
    run_dirs: list[Path],
    labels: list[str],
    output_path: Path,
    metrics: list[str] | None = None,
):
    """Plot learning curves comparing multiple runs."""
    if metrics is None:
        metrics = [
            "rollout/ep_rew_mean",
            "eval/mean_reward",
            "eval/success_rate",
            "train/actor_loss",
            "train/critic_loss",
            "train/ent_coef",
        ]

    # Load data from all runs
    all_data = []
    for run_dir in run_dirs:
        tb_dir = run_dir / "tensorboard" / "SAC_1"
        if tb_dir.exists():
            data = load_tensorboard_data(tb_dir)
            all_data.append(data)
        else:
            print(f"Warning: {tb_dir} not found")
            all_data.append({})

    # Filter to metrics that exist in at least one run
    available_metrics = []
    for metric in metrics:
        for data in all_data:
            if metric in data:
                available_metrics.append(metric)
                break

    if not available_metrics:
        print("No metrics found in tensorboard logs")
        return

    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    colors = plt.cm.tab10.colors

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        for i, (data, label) in enumerate(zip(all_data, labels)):
            if metric in data:
                steps = data[metric]["steps"]
                values = data[metric]["values"]
                if len(values) > 0:
                    # Plot raw data with transparency
                    ax.plot(steps, values, alpha=0.3, color=colors[i % len(colors)])
                    # Plot smoothed data
                    smoothed = smooth(values, weight=0.9)
                    ax.plot(
                        steps, smoothed, label=label, color=colors[i % len(colors)], linewidth=2
                    )

        ax.set_xlabel("Steps")
        ax.set_ylabel(metric.split("/")[-1])
        ax.set_title(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_single_run(run_dir: Path, output_path: Path | None = None):
    """Plot learning curves for a single run."""
    if output_path is None:
        output_path = run_dir / "learning_curves.png"

    tb_dir = run_dir / "tensorboard" / "SAC_1"
    if not tb_dir.exists():
        print(f"Tensorboard directory not found: {tb_dir}")
        return

    data = load_tensorboard_data(tb_dir)

    # Key metrics to plot
    metrics = [
        ("rollout/ep_rew_mean", "Episode Reward (Mean)"),
        ("eval/mean_reward", "Eval Reward"),
        ("eval/success_rate", "Success Rate"),
        ("train/ent_coef", "Entropy Coefficient"),
    ]

    available = [(m, title) for m, title in metrics if m in data]

    if not available:
        print(f"No metrics found in {tb_dir}")
        return

    n_metrics = len(available)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(available):
        ax = axes[idx]
        steps = data[metric]["steps"]
        values = data[metric]["values"]

        if len(values) > 0:
            ax.plot(steps, values, alpha=0.3, color="blue")
            smoothed = smooth(values, weight=0.9)
            ax.plot(steps, smoothed, color="blue", linewidth=2)

        ax.set_xlabel("Steps")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Hide unused
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    # Add run name as title
    fig.suptitle(f"Training: {run_dir.name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from tensorboard logs")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        help="Run directories to plot (can be multiple for comparison)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each run (must match number of runs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="learning_curves.png",
        help="Output file path",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple runs in one plot",
    )
    args = parser.parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        # Default: find all runs in lift_cube_cartesian
        runs_dir = Path("runs/lift_cube_cartesian")
        run_dirs = sorted(runs_dir.glob("*"))
        run_dirs = [r for r in run_dirs if r.is_dir()]

    if not run_dirs:
        print("No runs found")
        return

    if args.compare or len(run_dirs) > 1:
        labels = args.labels if args.labels else [r.name for r in run_dirs]
        if len(labels) != len(run_dirs):
            labels = [r.name for r in run_dirs]
        plot_comparison(run_dirs, labels, Path(args.output))
    else:
        plot_single_run(run_dirs[0], Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
