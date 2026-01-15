"""Top-down pick video capture with wide angle view.

Records the IK-based top-down pick motion for blog post.
"""
import mujoco
import numpy as np
import imageio
from pathlib import Path
from datetime import datetime
import argparse
from src.controllers.ik_controller import IKController

scene_path = Path("models/so101/lift_cube.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
gripperframe_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
static_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
moving_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")


def setup_camera(angle="wide"):
    """Setup camera for different viewing angles."""
    if angle == "wrist":
        return "wrist_cam"  # Use named camera from XML
    if angle == "wrist+wide":
        return ["wrist_cam", "wide"]  # Both views

    cam = mujoco.MjvCamera()

    if angle == "wide":
        # Wide angle - shows entire robot and workspace
        cam.lookat[:] = [0.20, 0.0, 0.08]
        cam.distance = 0.7
        cam.azimuth = 145
        cam.elevation = -25
    elif angle == "close":
        # Close-up on gripper and cube
        cam.lookat[:] = [0.25, 0.0, 0.05]
        cam.distance = 0.35
        cam.azimuth = 135
        cam.elevation = -20
    elif angle == "top":
        # Top-down view
        cam.lookat[:] = [0.25, 0.0, 0.05]
        cam.distance = 0.5
        cam.azimuth = 180
        cam.elevation = -60
    else:
        raise ValueError(f"Unknown angle: {angle}")

    return cam


def get_contacts():
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == cube_geom_id or g2 == cube_geom_id:
            other = g2 if g1 == cube_geom_id else g1
            contacts.append(other)
    return contacts


def is_grasping():
    contacts = get_contacts()
    has_static = static_pad_geom_id in contacts
    has_moving = moving_pad_geom_id in contacts
    return has_static and has_moving


def center_crop_square(img):
    """Center crop 640x480 to 480x480 (matches training preprocessing)."""
    h, w = img.shape[:2]
    crop_x = (w - h) // 2  # 80px from each side
    return img[:, crop_x:crop_x + h, :]


def record_topdown_pick(cube_pos, renderer, cam, frame_interval=5, scene_option=None):
    """Record top-down pick sequence, returns frames."""
    frames = []
    is_wrist = isinstance(cam, str) and cam == "wrist_cam"

    height_offset = 0.03
    gripper_open = 0.3
    gripper_closed = -0.8
    grasp_z_offset = 0.005
    finger_width_offset = -0.015
    locked_joints = [3, 4]

    # Initialize
    mujoco.mj_resetData(model, data)
    data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = cube_pos
    data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]
    data.qpos[3] = np.pi/2
    data.qpos[4] = np.pi/2
    data.ctrl[3] = np.pi/2
    data.ctrl[4] = np.pi/2
    mujoco.mj_forward(model, data)

    ik = IKController(model, data, end_effector_site="gripperframe")

    # Settle cube
    for _ in range(50):
        mujoco.mj_step(model, data)

    actual_cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr+3].copy()

    def capture():
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        frame = renderer.render().copy()
        if is_wrist:
            frame = center_crop_square(frame)
        return frame

    def run_steps(target_pos, gripper_action, steps, gain=0.5):
        for step in range(steps):
            ctrl = ik.step_toward_target(target_pos, gripper_action=gripper_action, gain=gain, locked_joints=locked_joints)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            if step % frame_interval == 0:
                frames.append(capture())

    # Initial frame
    frames.append(capture())
    print(f"Cube at: {actual_cube_pos}")

    # Step 1: Move above block
    print("1. Moving above...")
    above_pos = actual_cube_pos.copy()
    above_pos[2] += grasp_z_offset + height_offset
    above_pos[1] += finger_width_offset
    run_steps(above_pos, gripper_open, steps=300)

    # Step 2: Move down
    print("2. Moving down...")
    grasp_target = actual_cube_pos.copy()
    grasp_target[2] += grasp_z_offset
    grasp_target[1] += finger_width_offset
    run_steps(grasp_target, gripper_open, steps=200)

    # Step 3: Close gripper
    print("3. Closing gripper...")
    contact_step = None
    contact_action = None
    tighten_amount = 0.4
    grasp_action = gripper_closed

    for step in range(300):
        if contact_step is None:
            t = min(step / 250, 1.0)
            gripper = gripper_open - 2.0 * t
        else:
            steps_since = step - contact_step
            t_slow = min(steps_since / 100, 1.0)
            target_action = max(contact_action - tighten_amount, -1.0)
            gripper = contact_action + (target_action - contact_action) * t_slow

        ctrl = ik.step_toward_target(grasp_target, gripper_action=gripper, gain=0.5, locked_joints=locked_joints)
        ctrl[3] = np.pi/2
        ctrl[4] = np.pi/2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % frame_interval == 0:
            frames.append(capture())

        if is_grasping() and contact_step is None:
            contact_step = step
            contact_action = gripper
            print(f"   Contact at step {step}")

        if contact_step is not None:
            target_action = max(contact_action - tighten_amount, -1.0)
            if gripper <= target_action + 0.01:
                grasp_action = gripper
                break
    else:
        if contact_step is not None:
            grasp_action = gripper

    # Step 4: Lift
    print("4. Lifting...")
    lift_pos = actual_cube_pos.copy()
    lift_pos[2] += height_offset + 0.05

    for step in range(300):
        t = min(step / 200, 1.0)
        target = grasp_target + (lift_pos - grasp_target) * t

        ctrl = ik.step_toward_target(target, gripper_action=grasp_action, gain=0.3, locked_joints=locked_joints)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % frame_interval == 0:
            frames.append(capture())

    # Step 5: Hold
    print("5. Holding...")
    for step in range(150):
        ctrl = ik.step_toward_target(lift_pos, gripper_action=grasp_action, gain=0.3, locked_joints=locked_joints)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % frame_interval == 0:
            frames.append(capture())

    final_z = data.qpos[cube_qpos_addr + 2]
    success = final_z > actual_cube_pos[2] + 0.03
    print(f"\nFinal cube Z: {final_z:.4f}")
    print(f"Result: {'SUCCESS' if success else 'FAIL'}")

    return frames, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", choices=["wide", "close", "top", "wrist"], default="wide")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hide-sites", action="store_true", help="Hide site markers (e.g. camera marker)")
    args = parser.parse_args()

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = setup_camera(args.angle)

    # Scene options to hide sites if requested
    scene_option = None
    if args.hide_sites:
        scene_option = mujoco.MjvOption()
        # Hide all site groups (0-5)
        scene_option.sitegroup[:] = 0

    cube_pos = np.array([0.25, 0.0, 0.015])

    print(f"=== Top-Down Pick Video ({args.angle} angle) ===")
    frames, success = record_topdown_pick(cube_pos, renderer, cam, scene_option=scene_option)

    # Output path
    if args.output:
        video_path = Path(args.output)
    else:
        output_dir = Path("devlogs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = output_dir / f"topdown_pick_{args.angle}_{timestamp}.mp4"

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(video_path), frames, fps=args.fps)
    print(f"\nSaved {len(frames)} frames to {video_path}")

    renderer.close()


if __name__ == "__main__":
    main()
