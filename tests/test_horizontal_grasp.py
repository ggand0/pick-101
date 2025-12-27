"""Horizontal gripper IK test.

Tests whether a horizontal gripper orientation can work with pure IK.

Finding: The SO-101 arm geometry cannot reach table level (Z=0.015)
with a truly horizontal gripper. The current top-down approach
(wrist_flex=1.65) is necessary to reach the cube on the table.

This script demonstrates the limitation with a raised cube (on a platform).
"""
import mujoco
import mujoco.viewer
import numpy as np
import sys
import time
from pathlib import Path
from src.controllers.ik_controller import IKController

scene_path = Path("SO-ARM100/Simulation/SO101/lift_cube_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]
graspframe_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "graspframe")
static_fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "static_fingertip")
moving_fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "moving_fingertip")


def get_fingertip_midpoint():
    """Get midpoint between finger tips using the fingertip sites."""
    static_tip = data.site_xpos[static_fingertip_id].copy()
    moving_tip = data.site_xpos[moving_fingertip_id].copy()
    midpoint = (static_tip + moving_tip) / 2
    return midpoint


def get_finger_tips_world():
    """Get individual fingertip positions in world frame."""
    static_tip = data.site_xpos[static_fingertip_id].copy()
    moving_tip = data.site_xpos[moving_fingertip_id].copy()
    return static_tip, moving_tip


def add_marker_sphere(scn, pos, rgba, size=0.008):
    """Add a colored sphere marker at position."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0, 0], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.array(rgba, dtype=np.float32)
    )
    scn.ngeom += 1


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
    has_static = any(g in contacts for g in [27, 28])
    has_moving = any(g in contacts for g in [29, 30])
    return has_static and has_moving


def run_horizontal_grasp(cube_height=0.06, viewer=None):
    """Test horizontal gripper grasp at specified cube height."""
    mujoco.mj_resetData(model, data)

    cube_x, cube_y, cube_z = 0.25, 0.0, cube_height

    data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, cube_z]
    data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

    # Lean back shoulder_lift, rotate wrist 90 degrees for horizontal fingers
    data.qpos[1] = -1.0      # shoulder_lift leaned back
    data.qpos[4] = np.pi / 2 # wrist_roll 90 degrees
    data.ctrl[1] = -1.0
    data.ctrl[4] = np.pi / 2
    mujoco.mj_forward(model, data)

    ik = IKController(model, data, end_effector_site="graspframe")

    # Debug: print finger geom info
    print("\nFinger geom info:")
    for gid in [27, 28, 29, 30]:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        gtype = model.geom_type[gid]
        size = model.geom_size[gid]
        print(f"  {gid}: {name}, type={gtype}, size={size}")

    # Print initial position
    init_pos = ik.get_ee_position()
    print(f"\nInitial graspframe: {init_pos}")
    print(f"Initial joints: sh_pan={data.qpos[0]:.2f}, sh_lift={data.qpos[1]:.2f}, elbow={data.qpos[2]:.2f}, wrist_flex={data.qpos[3]:.2f}")

    def step_sim(n=1):
        for _ in range(n):
            mujoco.mj_step(model, data)
            if viewer:
                viewer.user_scn.ngeom = 0
                # Show fingertip positions (projected to tip level)
                static_tip, moving_tip = get_finger_tips_world()
                add_marker_sphere(viewer.user_scn, static_tip, [1, 0, 0, 1], size=0.008)  # static tip - red
                add_marker_sphere(viewer.user_scn, moving_tip, [0, 0, 1, 1], size=0.008)  # moving tip - blue
                # Show finger midpoint
                finger_mid = get_fingertip_midpoint()
                add_marker_sphere(viewer.user_scn, finger_mid, [0, 1, 0, 1], size=0.01)  # midpoint - green
                # Show graspframe for comparison
                graspframe = data.site_xpos[graspframe_site_id]
                add_marker_sphere(viewer.user_scn, graspframe, [1, 1, 0, 1], size=0.006)  # graspframe - yellow
                viewer.sync()

    # Let cube settle (falls due to gravity if spawned mid-air)
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Get actual cube position after settling
    actual_cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr+3].copy()
    print(f"\n=== Horizontal Grasp Test ===")
    print(f"Spawn Z: {cube_z}, Settled Z: {actual_cube_pos[2]:.3f}\n")

    # Step 0: Open gripper fully and let settle
    print("Opening gripper...")
    for _ in range(100):
        data.ctrl[5] = model.actuator_ctrlrange[5][1]  # fully open
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()

    # Check finger positions with open gripper
    finger_mid = get_fingertip_midpoint()
    static_tip, moving_tip = get_finger_tips_world()
    finger_gap = np.linalg.norm(static_tip - moving_tip)
    print(f"Finger gap: {finger_gap:.3f}m, Finger mid: {finger_mid}")

    # Step 1: Approach - target finger_midpoint to cube (not graspframe)
    # We compute where graspframe needs to be so that finger_mid lands on cube
    print("\nApproaching (targeting finger_mid to cube)...")
    for step in range(1000):
        # Where is finger_mid relative to graspframe?
        graspframe_pos = ik.get_ee_position()
        finger_mid = get_fingertip_midpoint()
        offset = finger_mid - graspframe_pos

        # Target graspframe such that finger_mid = cube_pos
        # finger_mid = graspframe + offset
        # cube_pos = graspframe + offset
        # graspframe = cube_pos - offset
        graspframe_target = actual_cube_pos - offset

        ctrl = ik.step_toward_target(graspframe_target, gripper_action=1.0, gain=0.5)
        data.ctrl[:] = ctrl
        step_sim()

        # Check how close finger_mid is to cube
        finger_mid = get_fingertip_midpoint()
        error = np.linalg.norm(actual_cube_pos - finger_mid)

        if step % 200 == 0:
            print(f"  step {step}: finger_mid error={error:.4f}")

        if error < 0.015:
            print(f"  Reached at step {step}")
            break

    # Final state
    graspframe_pos = ik.get_ee_position()
    finger_mid = get_fingertip_midpoint()

    print(f"\nGraspframe: {graspframe_pos}")
    print(f"Finger mid: {finger_mid}")
    print(f"Cube:       {actual_cube_pos}")
    print(f"Finger_mid-cube error: {np.linalg.norm(actual_cube_pos - finger_mid):.4f}")
    print(f"wrist_flex: {data.qpos[3]:.3f} (0=horizontal, 1.65=down)")

    # Step 2: Close (maintain finger_mid position)
    print("\nClosing gripper...")
    grasp_action = 1.0
    for step in range(500):
        t = min(step / 350, 1.0)
        gripper_action = 1.0 - 2.0 * t

        # Keep targeting finger_mid to cube as we close
        graspframe_pos = ik.get_ee_position()
        finger_mid = get_fingertip_midpoint()
        offset = finger_mid - graspframe_pos
        graspframe_target = actual_cube_pos - offset

        ctrl = ik.step_toward_target(graspframe_target, gripper_action=gripper_action, gain=0.5)
        data.ctrl[:] = ctrl
        step_sim()

        if is_grasping() and data.qpos[gripper_qpos_addr] < 0.25:
            grasp_action = gripper_action
            print(f"  Grasp at step {step}")
            break

    # Step 3: Lift (target finger_mid upward)
    print("Lifting...")
    lift_height = 0.08
    for step in range(500):
        t = min(step / 300, 1.0)
        lift_pos = actual_cube_pos + np.array([0, 0, lift_height * t])

        graspframe_pos = ik.get_ee_position()
        finger_mid = get_fingertip_midpoint()
        offset = finger_mid - graspframe_pos
        graspframe_target = lift_pos - offset

        ctrl = ik.step_toward_target(graspframe_target, gripper_action=grasp_action, gain=0.5)
        data.ctrl[:] = ctrl
        step_sim()

    # Step 4: Hold position
    print("Holding...")
    hold_pos = actual_cube_pos + np.array([0, 0, lift_height])
    for step in range(500):
        graspframe_pos = ik.get_ee_position()
        finger_mid = get_fingertip_midpoint()
        offset = finger_mid - graspframe_pos
        graspframe_target = hold_pos - offset

        ctrl = ik.step_toward_target(graspframe_target, gripper_action=grasp_action, gain=0.5)
        data.ctrl[:] = ctrl
        step_sim()

    final_z = data.qpos[cube_qpos_addr + 2]
    lifted = final_z > cube_z + 0.03
    print(f"\nCube Z: {final_z:.4f} (started at {cube_z})")
    print(f"Result: {'SUCCESS' if lifted else 'FAIL'}")

    return lifted


if __name__ == "__main__":
    use_viewer = "--viewer" in sys.argv

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Try with raised cube
            run_horizontal_grasp(cube_height=0.06, viewer=viewer)
            print("\nViewer open. Ctrl+C to exit.")
            while viewer.is_running():
                time.sleep(0.1)
    else:
        # Test at different heights
        print("Testing horizontal grasp at different cube heights:\n")
        for height in [0.015, 0.03, 0.05, 0.07, 0.09]:
            success = run_horizontal_grasp(cube_height=height)
            print()
