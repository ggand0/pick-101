"""Test wrist camera placement and rendering for image-based RL."""

import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.lift_cube import LiftCubeCartesianEnv


def test_wrist_camera_view(save_video: bool = True, num_steps: int = 200):
    """Test that wrist camera renders correctly and shows grasping area."""
    print("Creating environment...")
    env = LiftCubeCartesianEnv(
        render_mode="rgb_array",
        curriculum_stage=3,  # Gripper near cube
        reward_version="v11",
    )

    # Setup renderer for wrist camera (named camera from XML)
    wrist_renderer = mujoco.Renderer(env.model, height=84, width=84)
    # Also setup higher res for visualization
    wrist_renderer_hires = mujoco.Renderer(env.model, height=480, width=480)

    obs, info = env.reset()
    print(f"Environment reset. Cube pos: {info['cube_pos']}")

    frames_wrist = []
    frames_closeup = []
    frames_wide = []
    frames_combined = []

    print(f"Running {num_steps} steps with random actions...")
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Render from wrist camera (named camera from XML)
        wrist_renderer_hires.update_scene(env.data, camera="wrist_cam")
        wrist_frame = wrist_renderer_hires.render()

        # Render from external cameras using env's render method
        closeup_frame = env.render(camera="closeup")
        wide_frame = env.render(camera="wide")

        frames_wrist.append(wrist_frame)
        frames_closeup.append(closeup_frame)
        frames_wide.append(wide_frame)

        # Create 3-panel comparison: wide | closeup | wrist
        # All should be same height (480)
        target_h = 480

        # Resize wrist frame to match height
        wrist_h, wrist_w = wrist_frame.shape[:2]
        if wrist_h != target_h:
            scale = target_h / wrist_h
            new_w = int(wrist_w * scale)
            wrist_pil = Image.fromarray(wrist_frame)
            wrist_resized = np.array(wrist_pil.resize((new_w, target_h)))
        else:
            wrist_resized = wrist_frame

        # Resize wide and closeup if needed (they should already be 480 height)
        combined = np.concatenate([wide_frame, closeup_frame, wrist_resized], axis=1)
        frames_combined.append(combined)

        if step % 50 == 0:
            print(f"  Step {step}: cube_z={info['cube_z']:.3f}, grasping={info['is_grasping']}")

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            obs, info = env.reset()

    # Test low-res rendering (for actual RL training)
    wrist_renderer.update_scene(env.data, camera="wrist_cam")
    lowres_frame = wrist_renderer.render()
    print(f"\nLow-res wrist camera shape: {lowres_frame.shape}, dtype: {lowres_frame.dtype}")
    assert lowres_frame.shape == (84, 84, 3), f"Expected (84, 84, 3), got {lowres_frame.shape}"
    assert lowres_frame.dtype == np.uint8, f"Expected uint8, got {lowres_frame.dtype}"

    if save_video:
        output_dir = Path(__file__).parent.parent / "runs" / "camera_test"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save combined video
        video_path = output_dir / "wrist_camera_test.mp4"
        print(f"\nSaving video to {video_path}")
        imageio.mimwrite(str(video_path), frames_combined, fps=30)

        # Save sample frames
        for i, idx in enumerate([0, 50, 100, 150]):
            if idx < len(frames_wrist):
                frame_path = output_dir / f"wrist_frame_{idx:03d}.png"
                imageio.imwrite(str(frame_path), frames_wrist[idx])
                print(f"  Saved {frame_path}")

        # Save low-res sample
        lowres_path = output_dir / "wrist_lowres_84x84.png"
        imageio.imwrite(str(lowres_path), lowres_frame)
        print(f"  Saved {lowres_path}")

    env.close()
    print("\nCamera test completed successfully!")
    return True


def benchmark_rendering_fps(num_frames: int = 1000):
    """Benchmark wrist camera rendering speed."""
    import time

    print(f"\nBenchmarking rendering speed ({num_frames} frames)...")
    env = LiftCubeCartesianEnv(render_mode="rgb_array", curriculum_stage=3)
    renderer = mujoco.Renderer(env.model, height=84, width=84)

    obs, _ = env.reset()

    # Warmup
    for _ in range(10):
        renderer.update_scene(env.data, camera="wrist_cam")
        renderer.render()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_frames):
        action = env.action_space.sample()
        env.step(action)
        renderer.update_scene(env.data, camera="wrist_cam")
        renderer.render()
    elapsed = time.perf_counter() - start

    fps = num_frames / elapsed
    print(f"  Rendered {num_frames} frames in {elapsed:.2f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  Target for training: >50 FPS {'✓' if fps > 50 else '✗'}")

    env.close()
    return fps


if __name__ == "__main__":
    test_wrist_camera_view(save_video=True)
    benchmark_rendering_fps()
