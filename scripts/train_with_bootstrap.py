"""Train with bootstrapped replay buffer from scripted grasps.

Seeds the replay buffer with scripted trajectories before starting RL training.
This helps the agent discover successful grasps early.

Usage:
    # First collect trajectories:
    MUJOCO_GL=egl uv run python scripts/collect_scripted_grasps.py --episodes 500

    # Then train with bootstrap:
    MUJOCO_GL=egl uv run python scripts/train_with_bootstrap.py \
        --config configs/drqv2_lift_s3_v13.yaml \
        --bootstrap runs/bootstrap/scripted_grasps.pkl
"""

# Use spawn for multiprocessing (required for EGL/GPU rendering on AMD)
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import argparse
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.workspace import SO101Workspace
from src.training.config_loader import load_config, instantiate
from src.training.so101_factory import SO101Factory

# Monkey-patch hydra.utils.instantiate to use our version
import hydra.utils
hydra.utils.instantiate = instantiate


def load_bootstrap_trajectories(bootstrap_path: Path):
    """Load trajectories from pickle file."""
    with open(bootstrap_path, 'rb') as f:
        data = pickle.load(f)

    trajectories = data['trajectories']
    stats = data['stats']

    print(f"Loaded bootstrap data:")
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    print(f"  Mean reward: {stats['mean_reward']:.2f}")

    return trajectories, stats


def add_trajectories_to_buffer(workspace, trajectories, max_episodes: int = None):
    """Add trajectories to the replay buffer.

    Args:
        workspace: SO101Workspace with initialized replay buffer
        trajectories: list of trajectory dicts from collect_scripted_grasps.py
        max_episodes: limit number of episodes to add (None = all)
    """
    replay_buffer = workspace.replay_buffer

    if max_episodes is not None:
        trajectories = trajectories[:max_episodes]

    total_transitions = 0
    successful_episodes = 0

    for traj_data in trajectories:
        trajectory = traj_data['trajectory']
        success = traj_data['success']

        if success:
            successful_episodes += 1

        if len(trajectory) == 0:
            continue

        # Add each transition to replay buffer
        # RoboBase replay buffer expects: add(obs, action, reward, term, trunc)
        for i, trans in enumerate(trajectory):
            obs = trans['obs']
            action = trans['action']
            reward = trans['reward']
            terminated = trans['terminated']
            truncated = trans['truncated']

            try:
                replay_buffer.add(obs, action, reward, terminated, truncated)
                total_transitions += 1
            except Exception as e:
                print(f"Warning: Failed to add transition {i}: {e}")
                break

        # Add final observation for this episode
        final_obs = trajectory[-1]['obs']
        try:
            replay_buffer.add_final(final_obs)
        except Exception as e:
            print(f"Warning: Failed to add final obs: {e}")

    print(f"Added {total_transitions} transitions from {len(trajectories)} episodes")
    print(f"  Successful episodes: {successful_episodes}")

    return total_transitions


def main():
    parser = argparse.ArgumentParser(description="Train with bootstrapped replay buffer")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--bootstrap", type=str, required=True, help="Bootstrap pickle file")
    parser.add_argument("--max_bootstrap_episodes", type=int, default=None,
                        help="Max episodes to load from bootstrap")
    parser.add_argument("--resume", type=str, default=None, help="Resume from snapshot")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Load bootstrap trajectories
    bootstrap_path = Path(args.bootstrap)
    if not bootstrap_path.exists():
        print(f"Error: Bootstrap file not found: {bootstrap_path}")
        print("Run collect_scripted_grasps.py first.")
        sys.exit(1)

    trajectories, stats = load_bootstrap_trajectories(bootstrap_path)

    # Check success rate
    if stats['success_rate'] < 0.1:
        print(f"Warning: Low success rate ({stats['success_rate']*100:.1f}%)")
        print("Consider adjusting scripted policy or curriculum stage")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("experiment_name", "image_rl") + "_bootstrap"
    work_dir = Path("runs") / exp_name / timestamp
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output directory
    shutil.copy(args.config, work_dir / "config.yaml")

    # Copy bootstrap info
    shutil.copy(bootstrap_path, work_dir / "bootstrap_data.pkl")

    print(f"\nOutput directory: {work_dir}")

    # Create workspace
    workspace = SO101Workspace(
        cfg=cfg,
        env_factory=SO101Factory(),
        work_dir=str(work_dir),
    )

    # Add bootstrap trajectories to replay buffer
    print("\nSeeding replay buffer with bootstrap trajectories...")
    n_added = add_trajectories_to_buffer(
        workspace,
        trajectories,
        max_episodes=args.max_bootstrap_episodes
    )

    print(f"\nReplay buffer size after bootstrap: {len(workspace.replay_buffer)}")

    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        workspace.load_snapshot(Path(args.resume))

    # Train
    print("\nStarting training...")
    workspace.train()


if __name__ == "__main__":
    main()
