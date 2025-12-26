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

    # Print initial position
    init_pos = ik.get_ee_position()
    print(f"Initial graspframe: {init_pos}")
    print(f"Initial joints: sh_pan={data.qpos[0]:.2f}, sh_lift={data.qpos[1]:.2f}, elbow={data.qpos[2]:.2f}, wrist_flex={data.qpos[3]:.2f}")

    def step_sim(n=1):
        for _ in range(n):
            mujoco.mj_step(model, data)
            if viewer:
                viewer.sync()

    # Let cube settle (falls due to gravity if spawned mid-air)
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Get actual cube position after settling
    actual_cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr+3].copy()
    print(f"\n=== Horizontal Grasp Test ===")
    print(f"Spawn Z: {cube_z}, Settled Z: {actual_cube_pos[2]:.3f}\n")

    # Step 1: Approach - pure IK
    target = actual_cube_pos.copy()
    for step in range(800):
        ctrl = ik.step_toward_target(target, gripper_action=1.0, gain=0.5)
        data.ctrl[:] = ctrl
        step_sim()

        if step % 200 == 0:
            ee_pos = ik.get_ee_position()
            error = np.linalg.norm(target - ee_pos)
            print(f"  step {step}: error={error:.4f}")
            if error < 0.01:
                break

    ee_pos = ik.get_ee_position()
    print(f"\nGraspframe: {ee_pos}")
    print(f"Target:     {target}")
    print(f"Error:      {np.linalg.norm(target - ee_pos):.4f}")
    print(f"wrist_flex: {data.qpos[3]:.3f} (0=horizontal, 1.65=down)")

    # Step 2: Close
    print("\nClosing gripper...")
    grasp_action = 1.0
    for step in range(500):
        t = min(step / 350, 1.0)
        gripper_action = 1.0 - 2.0 * t
        ctrl = ik.step_toward_target(target, gripper_action=gripper_action, gain=0.5)
        data.ctrl[:] = ctrl
        step_sim()

        if is_grasping() and data.qpos[gripper_qpos_addr] < 0.25:
            grasp_action = gripper_action
            print(f"  Grasp at step {step}")
            break

    # Step 3: Lift
    print("Lifting...")
    for step in range(300):
        t = min(step / 200, 1.0)
        lift_target = np.array([cube_x, cube_y, cube_z + 0.06 * t])
        ctrl = ik.step_toward_target(lift_target, gripper_action=grasp_action, gain=0.5)
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
