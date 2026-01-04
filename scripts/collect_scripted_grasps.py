"""Collect scripted grasp trajectories for replay buffer bootstrapping.

Following QT-Opt approach: seed replay buffer with scripted policy trajectories
that have 15-30% success rate, so RL can learn from successful examples.

Usage:
    MUJOCO_GL=egl uv run python scripts/collect_scripted_grasps.py \
        --episodes 1000 \
        --output runs/bootstrap/scripted_grasps.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.lift_cube import LiftCubeCartesianEnv
from src.training.so101_factory import WristCameraWrapper


def scripted_grasp_policy(env, add_noise: bool = True, rng: np.random.Generator = None,
                          max_steps: int = 200, grasp_only: bool = False):
    """Execute scripted grasp sequence and collect trajectory.

    Phases (full lift):
    1. Descend with open gripper (25 steps)
    2. Close gripper (20 steps)
    3. Lift to target height (until cube_z >= 0.085)
    4. Hold at height (remaining steps to reach max_steps)

    Phases (grasp_only=True):
    1. Descend with open gripper (25 steps)
    2. Close gripper and hold (remaining steps)

    Returns:
        trajectory: list of (obs, action, reward, done, info) tuples
        success: bool - is_success for full lift, is_grasping for grasp_only
    """
    if rng is None:
        rng = np.random.default_rng()

    trajectory = []
    noise_scale = 0.2 if add_noise else 0.0
    target_z = 0.085  # Just above success threshold (0.08)
    reached_height = False
    step_count = 0

    def add_transition(obs, action, reward, terminated, truncated, info):
        trajectory.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': {k: v for k, v in info.items() if not isinstance(v, np.ndarray) or v.size < 10}
        })

    # Phase 1: Descend with open gripper (25 steps)
    for _ in range(25):
        if step_count >= max_steps:
            break
        action = np.array([
            rng.uniform(-0.2, 0.2) * noise_scale,
            rng.uniform(-0.2, 0.2) * noise_scale,
            -0.6 + rng.uniform(-0.1, 0.1) * noise_scale,
            0.8  # open gripper
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        add_transition(obs, action, reward, terminated, truncated, info)
        step_count += 1

        if terminated or truncated:
            if grasp_only:
                return trajectory, info.get('is_grasping', False)
            return trajectory, info.get('is_success', False)

    # Phase 2: Close gripper (20 steps for full, remaining for grasp_only)
    close_steps = max_steps - step_count if grasp_only else 20
    for _ in range(close_steps):
        if step_count >= max_steps:
            break
        action = np.array([
            rng.uniform(-0.1, 0.1) * noise_scale,
            rng.uniform(-0.1, 0.1) * noise_scale,
            rng.uniform(-0.1, 0.1) * noise_scale,
            -1.0  # close gripper
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        add_transition(obs, action, reward, terminated, truncated, info)
        step_count += 1

        if terminated or truncated:
            if grasp_only:
                return trajectory, info.get('is_grasping', False)
            return trajectory, info.get('is_success', False)

    # For grasp_only mode, we're done after closing
    if grasp_only:
        success = info.get('is_grasping', False)
        return trajectory, success

    # Phase 3: Lift until target height
    while step_count < max_steps and not reached_height:
        action = np.array([
            rng.uniform(-0.1, 0.1) * noise_scale,
            rng.uniform(-0.1, 0.1) * noise_scale,
            0.8 + rng.uniform(-0.1, 0.1) * noise_scale,  # lift
            -1.0  # keep closed
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        add_transition(obs, action, reward, terminated, truncated, info)
        step_count += 1

        cube_z = info.get('cube_z', 0)
        if cube_z >= target_z:
            reached_height = True

        if terminated or truncated:
            return trajectory, info.get('is_success', False)

    # Phase 4: Hold at height (remaining steps)
    while step_count < max_steps:
        # Stay in place with gripper closed
        action = np.array([
            rng.uniform(-0.05, 0.05) * noise_scale,  # minimal xy drift
            rng.uniform(-0.05, 0.05) * noise_scale,
            0.0,  # maintain height
            -1.0  # keep closed
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        add_transition(obs, action, reward, terminated, truncated, info)
        step_count += 1

        if terminated or truncated:
            return trajectory, info.get('is_success', False)

    success = info.get('is_success', False)
    return trajectory, success


def collect_trajectories(
    num_episodes: int,
    curriculum_stage: int = 3,
    image_size: int = 84,
    add_noise: bool = True,
    seed: int = 42,
    grasp_only: bool = False,
):
    """Collect trajectories using scripted policy.

    Args:
        grasp_only: If True, only descend and grasp (no lifting).
                   Success = is_grasping instead of is_success.

    Returns:
        all_trajectories: list of trajectories
        stats: dict with success rate, etc.
    """
    rng = np.random.default_rng(seed)

    # Create environment with image observations
    env = LiftCubeCartesianEnv(
        render_mode='rgb_array',
        curriculum_stage=curriculum_stage,
        reward_version='v13',
    )
    env = WristCameraWrapper(env, image_size=(image_size, image_size))

    all_trajectories = []
    successes = 0
    total_rewards = []

    desc = "Collecting grasp-only trajectories" if grasp_only else "Collecting trajectories"
    for ep in tqdm(range(num_episodes), desc=desc):
        # Reset with different seed each episode
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))

        # Collect initial observation
        trajectory, success = scripted_grasp_policy(
            env, add_noise=add_noise, rng=rng, grasp_only=grasp_only
        )

        # Prepend reset observation to trajectory
        if len(trajectory) > 0:
            trajectory[0]['obs_prev'] = obs

        all_trajectories.append({
            'trajectory': trajectory,
            'success': success,
            'episode': ep,
            'total_reward': sum(t['reward'] for t in trajectory),
        })

        if success:
            successes += 1
        total_rewards.append(sum(t['reward'] for t in trajectory))

    env.close()

    stats = {
        'num_episodes': num_episodes,
        'successes': successes,
        'success_rate': successes / num_episodes,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'curriculum_stage': curriculum_stage,
        'grasp_only': grasp_only,
    }

    return all_trajectories, stats


def main():
    parser = argparse.ArgumentParser(description="Collect scripted grasp trajectories")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--output", type=str, default="runs/bootstrap/scripted_grasps.pkl",
                        help="Output pickle file")
    parser.add_argument("--curriculum_stage", type=int, default=3, help="Curriculum stage")
    parser.add_argument("--image_size", type=int, default=84, help="Image size")
    parser.add_argument("--no_noise", action="store_true", help="Disable action noise")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grasp_only", action="store_true",
                        help="Only descend and grasp, no lifting. Success = is_grasping.")
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "grasp-only" if args.grasp_only else "full lift"
    print(f"Collecting {args.episodes} trajectories with scripted policy ({mode})...")
    print(f"  Curriculum stage: {args.curriculum_stage}")
    print(f"  Image size: {args.image_size}")
    print(f"  Action noise: {not args.no_noise}")
    print(f"  Grasp only: {args.grasp_only}")

    trajectories, stats = collect_trajectories(
        num_episodes=args.episodes,
        curriculum_stage=args.curriculum_stage,
        image_size=args.image_size,
        add_noise=not args.no_noise,
        seed=args.seed,
        grasp_only=args.grasp_only,
    )

    print(f"\nResults:")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Successes: {stats['successes']}/{stats['num_episodes']}")

    # Save
    data = {
        'trajectories': trajectories,
        'stats': stats,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved to {output_path}")

    # Also save a summary
    summary_path = output_path.with_suffix('.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Scripted Grasp Collection Summary\n")
        f.write(f"=================================\n\n")
        f.write(f"Mode: {'grasp-only' if stats.get('grasp_only') else 'full lift'}\n")
        f.write(f"Episodes: {stats['num_episodes']}\n")
        f.write(f"Success rate: {stats['success_rate']*100:.1f}%\n")
        f.write(f"Successes: {stats['successes']}\n")
        f.write(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n")
        f.write(f"Curriculum stage: {stats['curriculum_stage']}\n")

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
