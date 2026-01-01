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

            # Plot raw and smoothed
            ax.plot(steps, values, alpha=0.3, color='blue')
            ax.plot(steps, smooth(values, 0.9), color='blue', linewidth=2, label='Smoothed')

            ax.set_xlabel('Steps')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add final value annotation
            if len(values) > 0:
                final_val = smooth(values, 0.9)[-1]
                ax.annotate(f'{final_val:.2f}',
                           xy=(steps[-1], final_val),
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=10, color='blue')
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
        axes[0].plot(steps, values, alpha=0.3, color='blue')
        axes[0].plot(steps, smooth(values, 0.9), color='blue', linewidth=2, label='Train')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Reward')
        axes[0].grid(True, alpha=0.3)

        # Add eval rewards if available
        if 'eval/episode_reward' in data:
            eval_steps = data['eval/episode_reward']['steps']
            eval_values = data['eval/episode_reward']['values']
            axes[0].scatter(eval_steps, eval_values, color='red', s=50, zorder=5, label='Eval')
            axes[0].legend()

    # Success rate plot
    if 'train/episode_success' in data:
        steps = data['train/episode_success']['steps']
        values = data['train/episode_success']['values']
        axes[1].plot(steps, values, alpha=0.3, color='green')
        axes[1].plot(steps, smooth(values, 0.9), color='green', linewidth=2)
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Success Rate')
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)

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

            ax.plot(steps, values, alpha=0.3, color='purple')
            ax.plot(steps, smooth(values, 0.9), color='purple', linewidth=2)
            ax.set_xlabel('Steps')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
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
