"""Compare old vs new camera views for observation images.

Renders side-by-side comparison of:
- OLD camera: pos="0.0 -0.055 0.02" euler="0 0 3.14159" fovy="75"
- NEW camera: pos="0.02 -0.08 -0.06" euler="0.698 0 3.14159" fovy="103"
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

    # Reset environment
    env.reset(seed=args.seed)

    # Get camera ID
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

    # Save original camera settings
    orig_pos = env.model.cam_pos[cam_id].copy()
    orig_quat = env.model.cam_quat[cam_id].copy()
    orig_fovy = env.model.cam_fovy[cam_id]

    print(f"Current camera settings:")
    print(f"  pos: {orig_pos}")
    print(f"  quat: {orig_quat}")
    print(f"  fovy: {orig_fovy}")

    # Setup renderers
    renderer_84 = mujoco.Renderer(env.model, height=84, width=84)
    renderer_256 = mujoco.Renderer(env.model, height=256, width=256)

    # OLD camera settings (v13 working)
    old_pos = np.array([0.0, -0.055, 0.02])
    old_euler = np.array([0, 0, np.pi])  # 180° yaw
    old_quat = np.zeros(4)
    mujoco.mju_euler2Quat(old_quat, old_euler, 'xyz')
    old_fovy = 75

    # NEW camera settings (current - not working)
    new_pos = np.array([0.02, -0.08, -0.06])
    new_euler = np.array([0.698, 0, np.pi])  # 40° pitch + 180° yaw
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, new_euler, 'xyz')
    new_fovy = 103

    # Render with OLD camera
    env.model.cam_pos[cam_id] = old_pos
    env.model.cam_quat[cam_id] = old_quat
    env.model.cam_fovy[cam_id] = old_fovy
    mujoco.mj_forward(env.model, env.data)

    renderer_84.update_scene(env.data, camera="wrist_cam")
    old_img_84 = renderer_84.render()
    renderer_256.update_scene(env.data, camera="wrist_cam")
    old_img_256 = renderer_256.render()

    # Render with NEW camera
    env.model.cam_pos[cam_id] = new_pos
    env.model.cam_quat[cam_id] = new_quat
    env.model.cam_fovy[cam_id] = new_fovy
    mujoco.mj_forward(env.model, env.data)

    renderer_84.update_scene(env.data, camera="wrist_cam")
    new_img_84 = renderer_84.render()
    renderer_256.update_scene(env.data, camera="wrist_cam")
    new_img_256 = renderer_256.render()

    # Restore original settings
    env.model.cam_pos[cam_id] = orig_pos
    env.model.cam_quat[cam_id] = orig_quat
    env.model.cam_fovy[cam_id] = orig_fovy

    # Save individual images
    Image.fromarray(old_img_84).save(output_dir / "old_camera_84x84.png")
    Image.fromarray(old_img_256).save(output_dir / "old_camera_256x256.png")
    Image.fromarray(new_img_84).save(output_dir / "new_camera_84x84.png")
    Image.fromarray(new_img_256).save(output_dir / "new_camera_256x256.png")

    # Create side-by-side comparison
    combined_84 = Image.new('RGB', (84*2 + 10, 84), color=(128, 128, 128))
    combined_84.paste(Image.fromarray(old_img_84), (0, 0))
    combined_84.paste(Image.fromarray(new_img_84), (84+10, 0))
    combined_84.save(output_dir / "comparison_84x84.png")

    combined_256 = Image.new('RGB', (256*2 + 20, 256), color=(128, 128, 128))
    combined_256.paste(Image.fromarray(old_img_256), (0, 0))
    combined_256.paste(Image.fromarray(new_img_256), (256+20, 0))
    combined_256.save(output_dir / "comparison_256x256.png")

    print(f"\nOLD camera (v13 - worked):")
    print(f"  pos: {old_pos}")
    print(f"  euler: [0, 0, π]")
    print(f"  fovy: {old_fovy}°")

    print(f"\nNEW camera (current - not working):")
    print(f"  pos: {new_pos}")
    print(f"  euler: [0.698, 0, π]")
    print(f"  fovy: {new_fovy}°")

    print(f"\nImages saved to {output_dir}")
    print(f"  - old_camera_*.png: v13 camera settings (WORKED)")
    print(f"  - new_camera_*.png: current camera settings (NOT WORKING)")
    print(f"  - comparison_*.png: side-by-side (OLD | NEW)")

    renderer_84.close()
    renderer_256.close()
    env.close()


if __name__ == "__main__":
    main()
