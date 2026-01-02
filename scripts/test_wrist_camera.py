"""Test wrist camera configurations for sim-to-real alignment."""

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import time

OUTPUT_DIR = Path("runs/camera_test_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Original camera settings
ORIG_POS = [0.0, -0.055, 0.02]
ORIG_EULER = [0, 0, 180]  # degrees
ORIG_FOV = 75


def render_config_pair(name: str, pos: list, euler_deg: list, fov: float):
    """Render camera view + external view showing camera position as red sphere."""
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

    # Render wrist cam view
    renderer = mujoco.Renderer(model, height=256, width=256)
    renderer.update_scene(data, camera="wrist_cam")
    cam_view = renderer.render().copy()

    # Get camera world position for visualization
    cam_world_pos = data.cam_xpos[cam_id].copy()

    # Render external view showing camera position
    ext_cam = mujoco.MjvCamera()
    ext_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ext_cam.lookat[:] = cam_world_pos
    ext_cam.distance = 0.15
    ext_cam.azimuth = 135
    ext_cam.elevation = -30

    renderer.update_scene(data, camera=ext_cam)
    ext_view = renderer.render().copy()

    # Draw red circle on external view to mark camera position
    from PIL import ImageDraw
    ext_img = Image.fromarray(ext_view)
    draw = ImageDraw.Draw(ext_img)
    # Camera is at center of lookat, so draw circle in center
    cx, cy = 128, 128
    r = 8
    draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)], fill=(255, 0, 0), outline=(255, 255, 255))
    ext_view = np.array(ext_img)

    renderer.close()

    return cam_view, ext_view


def render_comparison_pairs(configs: list) -> np.ndarray:
    """Render configs as pairs: [cam_view | ext_view] for each config, stacked vertically."""
    rows = []
    for name, pos, euler_deg, fov in configs:
        cam_view, ext_view = render_config_pair(name, pos, euler_deg, fov)

        # Add labels
        from PIL import ImageDraw

        cam_img = Image.fromarray(cam_view)
        draw = ImageDraw.Draw(cam_img)
        draw.rectangle([(0, 0), (256, 20)], fill=(0, 0, 0))
        draw.text((5, 3), name[:30], fill=(255, 255, 255))

        ext_img = Image.fromarray(ext_view)
        draw = ImageDraw.Draw(ext_img)
        draw.rectangle([(0, 0), (256, 20)], fill=(0, 0, 0))
        draw.text((5, 3), "external", fill=(255, 255, 255))

        # Combine as row
        row = np.concatenate([np.array(cam_img), np.array(ext_img)], axis=1)
        rows.append(row)

    return np.concatenate(rows, axis=0)


def main():
    # Real camera: innoMaker 130° diagonal, 103° horizontal, ~120° vertical
    # For 1:1 square crop, effective FOV = horizontal = 103°
    REAL_FOV = 103

    # Test 1: Pitch -70 with camera closer to fingertips (more negative Z)
    # Fingertips are at Z ~ -0.104, graspframe at Z ~ -0.038
    print("Testing pitch -70 with Z positions (fov=103)...")
    configs_z = [
        ("p-70 z0.0", [0.0, -0.055, 0.0], [-70, 0, 180], REAL_FOV),
        ("p-70 z-0.01", [0.0, -0.055, -0.01], [-70, 0, 180], REAL_FOV),
        ("p-70 z-0.02", [0.0, -0.055, -0.02], [-70, 0, 180], REAL_FOV),
        ("p-70 z-0.03", [0.0, -0.055, -0.03], [-70, 0, 180], REAL_FOV),
    ]
    img = render_comparison_pairs(configs_z)
    Image.fromarray(img).save(OUTPUT_DIR / "pitch70_z_positions.png")
    print("Saved pitch70_z_positions.png")

    # Test 2: Pitch -70 with Y positions (forward/back - more negative = further forward)
    print("Testing pitch -70 with Y positions (fov=103)...")
    configs_y = [
        ("p-70 y-0.055", [0.0, -0.055, -0.02], [-70, 0, 180], REAL_FOV),
        ("p-70 y-0.065", [0.0, -0.065, -0.02], [-70, 0, 180], REAL_FOV),
        ("p-70 y-0.075", [0.0, -0.075, -0.02], [-70, 0, 180], REAL_FOV),
        ("p-70 y-0.085", [0.0, -0.085, -0.02], [-70, 0, 180], REAL_FOV),
    ]
    img = render_comparison_pairs(configs_y)
    Image.fromarray(img).save(OUTPUT_DIR / "pitch70_y_forward.png")
    print("Saved pitch70_y_forward.png")

    # Test 3: Fine-tune pitch around -70
    print("Testing pitch variations (fov=103)...")
    configs_pitch = [
        ("p-60 z-0.02", [0.0, -0.065, -0.02], [-60, 0, 180], REAL_FOV),
        ("p-70 z-0.02", [0.0, -0.065, -0.02], [-70, 0, 180], REAL_FOV),
        ("p-80 z-0.02", [0.0, -0.065, -0.02], [-80, 0, 180], REAL_FOV),
        ("p-90 z-0.02", [0.0, -0.065, -0.02], [-90, 0, 180], REAL_FOV),
    ]
    img = render_comparison_pairs(configs_pitch)
    Image.fromarray(img).save(OUTPUT_DIR / "pitch_variations.png")
    print("Saved pitch_variations.png")

    # Test 4: Combined - camera further forward with various pitches
    print("Testing combined configs (fov=103)...")
    configs_combo = [
        ("p-70 y-0.07 z-0.02", [0.0, -0.07, -0.02], [-70, 0, 180], REAL_FOV),
        ("p-75 y-0.07 z-0.025", [0.0, -0.07, -0.025], [-75, 0, 180], REAL_FOV),
        ("p-70 y-0.08 z-0.02", [0.0, -0.08, -0.02], [-70, 0, 180], REAL_FOV),
        ("p-75 y-0.08 z-0.025", [0.0, -0.08, -0.025], [-75, 0, 180], REAL_FOV),
    ]
    img = render_comparison_pairs(configs_combo)
    Image.fromarray(img).save(OUTPUT_DIR / "best_candidates.png")
    print("Saved best_candidates.png")


if __name__ == "__main__":
    main()
