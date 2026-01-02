"""Plot learning curves from tensorboard logs."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Installing tensorboard...")
    import subprocess
    subprocess.run(["uv", "pip", "install", "tensorboard"], check=True)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_data(log_dir: Path) -> dict:
    """Load all scalar data from tensorboard logs."""
    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': np.array(steps), 'values': np.array(values)}

    return data


def split_runs(steps: np.ndarray, values: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split data into separate runs based on step discontinuities.

    Detects when steps decrease or have large gaps, indicating a new run started.
    """
    if len(steps) == 0:
        return []

    runs = []
    run_start = 0

    for i in range(1, len(steps)):
        # Detect run boundary: step decreased or large gap (>2x previous delta)
        if steps[i] < steps[i-1]:
            # New run started (step reset)
            runs.append((steps[run_start:i], values[run_start:i]))
            run_start = i

    # Add final run
    if run_start < len(steps):
        runs.append((steps[run_start:], values[run_start:]))

    return runs


# Color palette for different runs
RUN_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
]


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i-1] + (1 - weight) * values[i]
    return smoothed


def plot_learning_curves(log_dir: Path, output_path: Path = None):
    """Plot learning curves from tensorboard logs."""
    data = load_tb_data(log_dir)

    # Print available tags
    print("Available metrics:")
    for tag in sorted(data.keys()):
        print(f"  - {tag}: {len(data[tag]['values'])} points")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DrQ-v2 Learning Curves (Stage 3 Curriculum)', fontsize=14)

    # Key metrics to plot (using actual tensorboard tag names)
    metrics = [
        ('train/episode_reward', 'Train Episode Reward', axes[0, 0]),
        ('train/episode_success', 'Train Success Rate', axes[0, 1]),
        ('eval/episode_reward', 'Eval Reward', axes[0, 2]),
        ('train/critic_loss', 'Critic Loss', axes[1, 0]),
        ('train/actor_loss', 'Actor Loss', axes[1, 1]),
        ('train/critic_q2', 'Q-Value', axes[1, 2]),
    ]

    for tag, title, ax in metrics:
        if tag in data:
            steps = data[tag]['steps']
            values = data[tag]['values']

            # Split into separate runs
            runs = split_runs(steps, values)

            for run_idx, (run_steps, run_values) in enumerate(runs):
                if len(run_steps) < 2:
                    continue
                color = RUN_COLORS[run_idx % len(RUN_COLORS)]
                label = f'Run {run_idx + 1}' if len(runs) > 1 else 'Smoothed'

                # Plot raw and smoothed
                ax.plot(run_steps, run_values, alpha=0.2, color=color)
                ax.plot(run_steps, smooth(run_values, 0.9), color=color, linewidth=2, label=label)

                # Add final value annotation for the last run
                if run_idx == len(runs) - 1 and len(run_values) > 0:
                    final_val = smooth(run_values, 0.9)[-1]
                    ax.annotate(f'{final_val:.2f}',
                               xy=(run_steps[-1], final_val),
                               xytext=(5, 0), textcoords='offset points',
                               fontsize=10, color=color)

            ax.set_xlabel('Steps')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if len(runs) > 1:
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for\n{tag}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.close()

    # Also create a summary plot with just reward and success
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training Summary', fontsize=14)

    # Reward plot
    if 'train/episode_reward' in data:
        steps = data['train/episode_reward']['steps']
        values = data['train/episode_reward']['values']
        runs = split_runs(steps, values)

        for run_idx, (run_steps, run_values) in enumerate(runs):
            if len(run_steps) < 2:
                continue
            color = RUN_COLORS[run_idx % len(RUN_COLORS)]
            label = f'Run {run_idx + 1}' if len(runs) > 1 else 'Train'
            axes[0].plot(run_steps, run_values, alpha=0.2, color=color)
            axes[0].plot(run_steps, smooth(run_values, 0.9), color=color, linewidth=2, label=label)

        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Reward')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # Success rate plot
    if 'train/episode_success' in data:
        steps = data['train/episode_success']['steps']
        values = data['train/episode_success']['values']
        runs = split_runs(steps, values)

        for run_idx, (run_steps, run_values) in enumerate(runs):
            if len(run_steps) < 2:
                continue
            color = RUN_COLORS[run_idx % len(RUN_COLORS)]
            label = f'Run {run_idx + 1}' if len(runs) > 1 else None
            axes[1].plot(run_steps, run_values, alpha=0.2, color=color)
            axes[1].plot(run_steps, smooth(run_values, 0.9), color=color, linewidth=2, label=label)

        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Success Rate')
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)
        if len(runs) > 1:
            axes[1].legend()

    plt.tight_layout()

    if output_path:
        summary_path = output_path.parent / f"{output_path.stem}_summary{output_path.suffix}"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to {summary_path}")

    plt.close()

    # Task-specific metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Task Progress Metrics', fontsize=14)

    task_metrics = [
        ('train/env_info/gripper_to_cube', 'Gripper-to-Cube Distance', axes[0, 0]),
        ('train/env_info/cube_z', 'Cube Z Position', axes[0, 1]),
        ('train/env_info/is_grasping', 'Grasping Rate', axes[1, 0]),
        ('train/env_info/is_lifted', 'Lift Rate', axes[1, 1]),
    ]

    for tag, title, ax in task_metrics:
        if tag in data:
            steps = data[tag]['steps']
            values = data[tag]['values']
            runs = split_runs(steps, values)

            for run_idx, (run_steps, run_values) in enumerate(runs):
                if len(run_steps) < 2:
                    continue
                color = RUN_COLORS[run_idx % len(RUN_COLORS)]
                label = f'Run {run_idx + 1}' if len(runs) > 1 else None
                ax.plot(run_steps, run_values, alpha=0.2, color=color)
                ax.plot(run_steps, smooth(run_values, 0.9), color=color, linewidth=2, label=label)

            ax.set_xlabel('Steps')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if len(runs) > 1:
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for\n{tag}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    plt.tight_layout()

    if output_path:
        task_path = output_path.parent / f"{output_path.stem}_task{output_path.suffix}"
        plt.savefig(task_path, dpi=150, bbox_inches='tight')
        print(f"Saved task metrics plot to {task_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='runs/image_rl/tb_logs/drqv2_lift_s3')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = log_dir.parent.parent / 'learning_curves.png'

    plot_learning_curves(log_dir, output_path)


if __name__ == '__main__':
    main()
