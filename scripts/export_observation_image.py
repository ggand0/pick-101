"""Export the actual 84x84px observation images the agent sees.

Renders the wrist_cam view at different timesteps during an episode.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.lift_cube import LiftCubeCartesianEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for images")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 10, 50, 100],
                        help="Timesteps to capture (default: 0, 10, 50, 100)")
    parser.add_argument("--curriculum_stage", type=int, default=3, help="Curriculum stage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = LiftCubeCartesianEnv(
        render_mode=None,
        curriculum_stage=args.curriculum_stage,
        reward_version="v16",
    )

    # Setup 84x84 renderer for wrist_cam
    renderer = mujoco.Renderer(env.model, height=84, width=84)

    # Also create a larger renderer for visualization
    renderer_large = mujoco.Renderer(env.model, height=256, width=256)

    # Reset environment
    env.reset(seed=args.seed)

    print(f"Exporting observation images to {output_dir}")
    print(f"Capturing at steps: {args.steps}")

    step = 0
    for target_step in sorted(args.steps):
        # Advance to target step
        while step < target_step:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break

        if step > max(args.steps):
            break

        # Render 84x84 observation (what agent sees)
        renderer.update_scene(env.data, camera="wrist_cam")
        img_84 = renderer.render()

        # Render 256x256 for easier viewing
        renderer_large.update_scene(env.data, camera="wrist_cam")
        img_256 = renderer_large.render()

        # Save images
        img_84_path = output_dir / f"obs_84x84_step{step:03d}.png"
        img_256_path = output_dir / f"obs_256x256_step{step:03d}.png"

        Image.fromarray(img_84).save(img_84_path)
        Image.fromarray(img_256).save(img_256_path)

        print(f"  Step {step}: saved {img_84_path.name} and {img_256_path.name}")

        # Print info
        info = env._get_info()
        print(f"    Cube Z: {info['cube_z']:.4f}, Gripper: {info['gripper_state']:.3f}, "
              f"Distance: {info['gripper_to_cube']:.4f}, Grasping: {info['is_grasping']}")

    renderer.close()
    renderer_large.close()
    env.close()

    print(f"\nDone! Images saved to {output_dir}")


if __name__ == "__main__":
    main()
