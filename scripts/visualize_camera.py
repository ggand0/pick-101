"""Visualize wrist camera and other views for SO-101 arm."""

import argparse
from pathlib import Path
import numpy as np
import imageio
import mujoco


def add_camera_markers(scene, cam_world_pos: np.ndarray, cam_dir: np.ndarray):
    """Add 3D markers to scene showing camera position and view direction."""
    # Target point 5cm along view direction
    view_target = cam_world_pos + cam_dir * 0.05
    line_center = (cam_world_pos + view_target) / 2
    line_length = np.linalg.norm(view_target - cam_world_pos)

    # Rotation matrix to align cylinder with direction
    z_axis = cam_dir / np.linalg.norm(cam_dir)
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
    else:
        x_axis = np.cross(np.array([1, 0, 0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])

    # Red sphere at camera position
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.008, 0, 0]),
        cam_world_pos,
        np.eye(3).flatten(),
        np.array([1, 0, 0, 1]),
    )
    scene.ngeom += 1

    # Green sphere at view target
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.006, 0, 0]),
        view_target,
        np.eye(3).flatten(),
        np.array([0, 1, 0, 1]),
    )
    scene.ngeom += 1

    # Green cylinder connecting them
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.003, line_length / 2, 0]),
        line_center,
        rot_mat.flatten(),
        np.array([0, 1, 0, 0.8]),
    )
    scene.ngeom += 1


def render_wrist_cam_4x3_cropped(model, data) -> np.ndarray:
    """Render wrist_cam at 640x480 (4:3) and center-crop to 480x480 square.

    This simulates the real camera pipeline: 640x480 -> center crop to 480x480.
    """
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="wrist_cam")
    frame = renderer.render().copy()
    renderer.close()

    # Center crop to 480x480 (crop 80px from each side)
    crop_x = (640 - 480) // 2
    cropped = frame[:, crop_x:crop_x + 480, :]

    return cropped


def render_multi_camera(model, data, cameras: list[str], size: int = 256, cam_id: int = None) -> dict[str, np.ndarray]:
    """Render from multiple camera views."""
    frames = {}
    renderer = mujoco.Renderer(model, height=size, width=size)

    # Get wrist camera world position and direction for markers
    cam_world_pos = None
    cam_dir = None
    if cam_id is not None:
        cam_world_pos = data.cam_xpos[cam_id].copy()
        cam_world_mat = data.cam_xmat[cam_id].reshape(3, 3)
        cam_dir = -cam_world_mat[:, 2]  # Camera looks along -Z

    # Virtual camera configurations: (lookat, distance, azimuth, elevation)
    virtual_cameras = {
        "topdown": ([0.35, 0.0, 0.0], 0.8, 90, -90),
        "side": ([0.35, 0.0, 0.05], 0.8, 90, -15),
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
            # Add camera markers to external views
            if cam_world_pos is not None and cam_dir is not None:
                add_camera_markers(renderer._scene, cam_world_pos, cam_dir)
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


def apply_calibrated_camera(model):
    """Apply calibrated wrist camera settings (from devlog 043)."""
    from scipy.spatial.transform import Rotation as R

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

    # Calibrated settings (v5): B=6.5cm, pitch=-22.8°
    model.cam_pos[cam_id] = [0.008, -0.065, -0.019]
    model.cam_fovy[cam_id] = 70.5

    rot = R.from_euler("xyz", [-22.8, 0, 180], degrees=True)
    q = rot.as_quat()  # x,y,z,w
    model.cam_quat[cam_id] = [q[3], q[0], q[1], q[2]]  # w,x,y,z

    return cam_id


def main():
    parser = argparse.ArgumentParser(description="Visualize SO-101 camera views")
    parser.add_argument("--output", type=str, default="runs/camera_test_v5",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="models/so101/lift_cube_calibration.xml",
                        help="MuJoCo model path")
    parser.add_argument("--size", type=int, default=256, help="Frame size")
    parser.add_argument("--video", action="store_true", help="Generate video")
    parser.add_argument("--steps", type=int, default=100, help="Video steps")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = Path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Apply calibrated camera settings
    cam_id = apply_calibrated_camera(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"Model: {model_path}")

    # Camera info
    print(f"wrist_cam position: {model.cam_pos[cam_id]}")
    print(f"wrist_cam fovy: {model.cam_fovy[cam_id]}")

    # Render static views
    cameras = ["wrist_cam", "topdown", "side", "iso"]
    frames = render_multi_camera(model, data, cameras, args.size, cam_id=cam_id)

    for cam_name, frame in frames.items():
        path = output_dir / f"{cam_name}.png"
        imageio.imwrite(str(path), frame)
        print(f"Saved {path}")

    # 4:3 cropped version (simulates real camera pipeline: 640x480 -> 480x480)
    cropped = render_wrist_cam_4x3_cropped(model, data)
    cropped_path = output_dir / "wrist_cam_4x3_cropped.png"
    imageio.imwrite(str(cropped_path), cropped)
    print(f"Saved {cropped_path}")

    # Combined grid
    combined = combine_frames(frames, layout="grid")
    combined_path = output_dir / "combined_grid.png"
    imageio.imwrite(str(combined_path), combined)
    print(f"Saved {combined_path}")

    # Render gripper poses (closed, half-open, fully-open)
    gripper_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_poses = {
        "closed": -0.17,
        "half_open": 0.78,
        "fully_open": 1.74,
    }
    print("\nRendering gripper poses...")
    for pose_name, ctrl_val in gripper_poses.items():
        mujoco.mj_resetData(model, data)
        data.ctrl[gripper_idx] = ctrl_val
        for _ in range(100):
            mujoco.mj_step(model, data)
        pose_frames = render_multi_camera(model, data, ["wrist_cam"], args.size, cam_id=cam_id)
        pose_path = output_dir / f"gripper_{pose_name}.png"
        imageio.imwrite(str(pose_path), pose_frames["wrist_cam"])
        print(f"Saved {pose_path}")

    # Create video with random motion (optional)
    if args.video:
        print(f"\nCreating video with {args.steps} steps...")
        mujoco.mj_resetData(model, data)

        video_frames = []
        for step in range(args.steps):
            # Random control
            ctrl = np.random.uniform(-0.3, 0.3, size=model.nu)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            cam_frames = render_multi_camera(model, data, cameras, args.size, cam_id=cam_id)
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
