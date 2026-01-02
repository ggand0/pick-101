"""Test wrist camera configurations for sim-to-real alignment."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from pathlib import Path

OUTPUT_DIR = Path("runs/camera_test_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Real camera: innoMaker 130° diagonal, 103° horizontal, ~120° vertical
# For 1:1 square crop, effective FOV = horizontal = 103°
REAL_FOV = 103


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


def render_triple_view(name: str, pos: list, euler_deg: list, fov: float):
    """Render: camera view | isometric with 3D markers | top-down with 3D markers."""
    model = mujoco.MjModel.from_xml_path("models/so101/lift_cube.xml")
    data = mujoco.MjData(model)

    # Set gripper closed
    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    data.qpos[gripper_joint_id] = 0.6

    # Arm reaching pose
    for jname, jval in zip(
        ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        [0.0, 0.8, -1.2, 0.8, 0.0],
    ):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[jid] = jval

    mujoco.mj_forward(model, data)

    # Apply camera settings
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    model.cam_pos[cam_id] = np.array(pos)
    model.cam_fovy[cam_id] = fov

    rot = R.from_euler("xyz", euler_deg, degrees=True)
    q = rot.as_quat()  # x,y,z,w
    model.cam_quat[cam_id] = [q[3], q[0], q[1], q[2]]  # w,x,y,z

    mujoco.mj_forward(model, data)

    # Get camera world position and direction
    cam_world_pos = data.cam_xpos[cam_id].copy()
    cam_world_mat = data.cam_xmat[cam_id].reshape(3, 3)
    cam_dir = -cam_world_mat[:, 2]  # Camera looks along -Z
    view_target = cam_world_pos + cam_dir * 0.05

    renderer = mujoco.Renderer(model, height=256, width=256)

    # 1. WRIST CAM view
    renderer.update_scene(data, camera="wrist_cam")
    cam_view = renderer.render().copy()

    # 2. ISOMETRIC view with 3D markers
    iso_cam = mujoco.MjvCamera()
    iso_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    midpoint = (cam_world_pos + view_target) / 2
    iso_cam.lookat[:] = midpoint
    iso_cam.distance = 0.12
    iso_cam.azimuth = 135
    iso_cam.elevation = -30

    renderer.update_scene(data, camera=iso_cam)
    add_camera_markers(renderer._scene, cam_world_pos, cam_dir)
    iso_view = renderer.render().copy()

    # 3. TOP-DOWN view with 3D markers
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    gripper_pos = data.xpos[gripper_body_id].copy()

    top_cam = mujoco.MjvCamera()
    top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    top_cam.lookat[:] = gripper_pos
    top_cam.distance = 0.25
    top_cam.azimuth = 180
    top_cam.elevation = -89

    renderer.update_scene(data, camera=top_cam)
    add_camera_markers(renderer._scene, cam_world_pos, cam_dir)
    top_view = renderer.render().copy()

    renderer.close()
    return cam_view, iso_view, top_view


def render_comparison_grid(configs: list, filename: str):
    """Render configs as rows: [cam_view | iso_view | top_view]."""
    rows = []
    for name, pos, euler_deg, fov in configs:
        cam_view, iso_view, top_view = render_triple_view(name, pos, euler_deg, fov)

        cam_img = Image.fromarray(cam_view)
        draw = ImageDraw.Draw(cam_img)
        draw.rectangle([(0, 0), (256, 20)], fill=(0, 0, 0))
        draw.text((5, 3), name[:30], fill=(255, 255, 255))

        iso_img = Image.fromarray(iso_view)
        draw = ImageDraw.Draw(iso_img)
        draw.rectangle([(0, 0), (256, 20)], fill=(0, 0, 0))
        draw.text((5, 3), "isometric", fill=(255, 255, 255))

        top_img = Image.fromarray(top_view)
        draw = ImageDraw.Draw(top_img)
        draw.rectangle([(0, 0), (256, 20)], fill=(0, 0, 0))
        draw.text((5, 3), "top-down", fill=(255, 255, 255))

        row = np.concatenate(
            [np.array(cam_img), np.array(iso_img), np.array(top_img)], axis=1
        )
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(OUTPUT_DIR / filename)
    print(f"Saved {filename}")


def main():
    # Test Z positions (closer to fingertips)
    print("001: Z positions (camera closer to fingertips)...")
    configs_z = [
        ("z=-0.06", [0.0, -0.055, -0.06], [-70, 0, 180], REAL_FOV),
        ("z=-0.07", [0.0, -0.055, -0.07], [-70, 0, 180], REAL_FOV),
        ("z=-0.08", [0.0, -0.055, -0.08], [-70, 0, 180], REAL_FOV),
        ("z=-0.09", [0.0, -0.055, -0.09], [-70, 0, 180], REAL_FOV),
    ]
    render_comparison_grid(configs_z, "001_z_positions.png")

    # Test pitch variations
    print("002: Pitch variations at z=-0.08...")
    configs_pitch = [
        ("p-50 z-0.08", [0.0, -0.055, -0.08], [-50, 0, 180], REAL_FOV),
        ("p-60 z-0.08", [0.0, -0.055, -0.08], [-60, 0, 180], REAL_FOV),
        ("p-70 z-0.08", [0.0, -0.055, -0.08], [-70, 0, 180], REAL_FOV),
        ("p-80 z-0.08", [0.0, -0.055, -0.08], [-80, 0, 180], REAL_FOV),
    ]
    render_comparison_grid(configs_pitch, "002_pitch_variations.png")

    # Test Y positions (forward/back)
    print("003: Y positions...")
    configs_y = [
        ("y=-0.055", [0.0, -0.055, -0.08], [-70, 0, 180], REAL_FOV),
        ("y=-0.07", [0.0, -0.07, -0.08], [-70, 0, 180], REAL_FOV),
        ("y=-0.085", [0.0, -0.085, -0.08], [-70, 0, 180], REAL_FOV),
        ("y=-0.10", [0.0, -0.10, -0.08], [-70, 0, 180], REAL_FOV),
    ]
    render_comparison_grid(configs_y, "003_y_positions.png")

    # Best candidates
    print("004: Best candidates...")
    configs_best = [
        ("p-60 y-0.08 z-0.08", [0.0, -0.08, -0.08], [-60, 0, 180], REAL_FOV),
        ("p-65 y-0.08 z-0.08", [0.0, -0.08, -0.08], [-65, 0, 180], REAL_FOV),
        ("p-70 y-0.085 z-0.08", [0.0, -0.085, -0.08], [-70, 0, 180], REAL_FOV),
        ("p-75 y-0.09 z-0.08", [0.0, -0.09, -0.08], [-75, 0, 180], REAL_FOV),
    ]
    render_comparison_grid(configs_best, "004_best_candidates.png")


if __name__ == "__main__":
    main()
