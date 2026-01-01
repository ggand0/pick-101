"""Visualize wrist camera and other views for SO-101 arm."""

import argparse
from pathlib import Path
import numpy as np
import imageio
import mujoco


def render_multi_camera(model, data, cameras: list[str], size: int = 256) -> dict[str, np.ndarray]:
    """Render from multiple camera views."""
    frames = {}
    renderer = mujoco.Renderer(model, height=size, width=size)

    # Virtual camera configurations: (lookat, distance, azimuth, elevation)
    virtual_cameras = {
        "topdown": ([0.35, 0.0, 0.0], 0.8, 90, -90),
        "side": ([0.35, 0.0, 0.1], 0.7, 0, -15),
        "front": ([0.35, 0.0, 0.1], 0.7, 90, -15),
        "iso": ([0.35, 0.0, 0.1], 0.8, 135, -30),
    }

    for cam in cameras:
        if cam in virtual_cameras:
            lookat, dist, azim, elev = virtual_cameras[cam]
            cam_obj = mujoco.MjvCamera()
            cam_obj.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam_obj.lookat[:] = lookat
            cam_obj.distance = dist
            cam_obj.azimuth = azim
            cam_obj.elevation = elev
            renderer.update_scene(data, camera=cam_obj)
        else:
            try:
                renderer.update_scene(data, camera=cam)
            except Exception:
                print(f"Warning: Camera '{cam}' not found")
                continue

        frames[cam] = renderer.render().copy()

    renderer.close()
    return frames


def combine_frames(frames: dict[str, np.ndarray], layout: str = "grid") -> np.ndarray:
    """Combine multiple camera frames into a single image."""
    frame_list = list(frames.values())
    if len(frame_list) == 0:
        raise ValueError("No frames to combine")

    if layout == "grid" and len(frame_list) == 4:
        top = np.concatenate(frame_list[:2], axis=1)
        bottom = np.concatenate(frame_list[2:], axis=1)
        return np.concatenate([top, bottom], axis=0)
    else:
        return np.concatenate(frame_list, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Visualize SO-101 camera views")
    parser.add_argument("--output", type=str, default="runs/camera_test_v2",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="models/so101/lift_cube.xml",
                        help="MuJoCo model path")
    parser.add_argument("--size", type=int, default=256, help="Frame size")
    parser.add_argument("--steps", type=int, default=100, help="Video steps")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = Path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"Model: {model_path}")

    # Camera info
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    print(f"wrist_cam position: {model.cam_pos[cam_id]}")
    print(f"  Y=-0.055 means front side (away from arm base)")

    # Render static views
    cameras = ["wrist_cam", "topdown", "side", "iso"]
    frames = render_multi_camera(model, data, cameras, args.size)

    for cam_name, frame in frames.items():
        path = output_dir / f"{cam_name}.png"
        imageio.imwrite(str(path), frame)
        print(f"Saved {path}")

    # Combined grid
    combined = combine_frames(frames, layout="grid")
    combined_path = output_dir / "combined_grid.png"
    imageio.imwrite(str(combined_path), combined)
    print(f"Saved {combined_path}")

    # Create video with random motion
    print(f"\nCreating video with {args.steps} steps...")
    mujoco.mj_resetData(model, data)

    video_frames = []
    for step in range(args.steps):
        # Random control
        ctrl = np.random.uniform(-0.3, 0.3, size=model.nu)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        cam_frames = render_multi_camera(model, data, cameras, args.size)
        combined = combine_frames(cam_frames, layout="grid")
        video_frames.append(combined)

    video_path = output_dir / "camera_demo.mp4"
    writer = imageio.get_writer(str(video_path), fps=30, codec="libx264", quality=8)
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved {video_path}")

    print(f"\nOutput files in {output_dir}/")


if __name__ == "__main__":
    main()
