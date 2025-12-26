"""Test IK grasp with side approach (horizontal gripper).

Compares two approaches:
1. Top-down: wrist_flex=1.65 (pointing down), approach from above
2. Side: wrist_flex=0 (horizontal), approach from the side

The challenge: can the arm reach cube height (z=0.015) with horizontal gripper?

Run with --viewer for interactive visualization.
"""
import mujoco
import mujoco.viewer
import numpy as np
import sys
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


def get_finger_mid():
    """Get midpoint between gripper fingers."""
    f28 = data.geom_xpos[28]
    f30 = data.geom_xpos[30]
    return (f28 + f30) / 2


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


def run_grasp(wrist_flex, wrist_roll, approach="top", viewer=None):
    """Run a grasp attempt with given wrist configuration.

    Args:
        wrist_flex: Wrist flex angle (0=horizontal, 1.65=pointing down)
        wrist_roll: Wrist roll angle (0=default, pi/2=horizontal fingers)
        approach: "top" for top-down, "side" for side approach
        viewer: Optional viewer for visualization
    """
    mujoco.mj_resetData(model, data)

    # Cube position
    cube_x, cube_y, cube_z = 0.32, 0.0, 0.015

    data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, cube_z]
    data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

    # Set initial wrist configuration
    data.qpos[3] = wrist_flex
    data.qpos[4] = wrist_roll
    data.ctrl[3] = wrist_flex
    data.ctrl[4] = wrist_roll
    mujoco.mj_forward(model, data)

    ik = IKController(model, data, end_effector_site="gripperframe")

    def step_sim(n=1):
        for _ in range(n):
            mujoco.mj_step(model, data)
            if viewer:
                viewer.sync()

    print(f"\n=== Grasp Test: wrist_flex={wrist_flex:.2f}, wrist_roll={wrist_roll:.2f}, approach={approach} ===\n")

    # Step 1: Approach
    if approach == "top":
        # Position above cube first
        print("Step 1: Position above cube...")
        for step in range(400):
            target = np.array([cube_x, cube_y, cube_z + 0.05])
            ctrl = ik.step_toward_target(target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
            ctrl[3] = wrist_flex
            ctrl[4] = wrist_roll
            data.ctrl[:] = ctrl
            step_sim()

        # Descend
        print("Step 2: Descend to cube...")
        for step in range(400):
            target = np.array([cube_x, cube_y, cube_z])
            ctrl = ik.step_toward_target(target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
            ctrl[3] = wrist_flex
            ctrl[4] = wrist_roll
            data.ctrl[:] = ctrl
            step_sim()
    else:
        # Side approach: start from -X direction
        print("Step 1: Approach from side...")
        for step in range(400):
            target = np.array([cube_x - 0.06, cube_y, cube_z])
            ctrl = ik.step_toward_target(target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
            ctrl[3] = wrist_flex
            ctrl[4] = wrist_roll
            data.ctrl[:] = ctrl
            step_sim()

        print("Step 2: Move to cube...")
        for step in range(400):
            target = np.array([cube_x, cube_y, cube_z])
            ctrl = ik.step_toward_target(target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
            ctrl[3] = wrist_flex
            ctrl[4] = wrist_roll
            data.ctrl[:] = ctrl
            step_sim()

    # Check position
    ee_pos = ik.get_ee_position()
    finger_mid = get_finger_mid()
    cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr+3]
    print(f"  Gripperframe: {ee_pos}")
    print(f"  Finger mid:   {finger_mid}")
    print(f"  Cube:         {cube_pos}")
    print(f"  Finger-cube Z delta: {finger_mid[2] - cube_pos[2]:.4f}")

    # Step 3: Close gripper
    print("\nStep 3: Closing gripper...")
    grasp_action = 1.0
    for step in range(600):
        t = min(step / 400, 1.0)
        gripper_action = 1.0 - 2.0 * t

        target = np.array([cube_x, cube_y, cube_z])
        ctrl = ik.step_toward_target(target, gripper_action=gripper_action, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = wrist_flex
        ctrl[4] = wrist_roll
        data.ctrl[:] = ctrl
        step_sim()

        gripper_pos = data.qpos[gripper_qpos_addr]
        if is_grasping() and gripper_pos < 0.25:
            grasp_action = gripper_action
            print(f"  Grasp achieved at step {step}, gripper={gripper_pos:.3f}")
            break

    # Step 4: Lift
    print("\nStep 4: Lifting...")
    for step in range(300):
        t = min(step / 200, 1.0)
        lift_z = cube_z + 0.08 * t
        target = np.array([cube_x, cube_y, lift_z])
        ctrl = ik.step_toward_target(target, gripper_action=grasp_action, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = wrist_flex
        ctrl[4] = wrist_roll
        data.ctrl[:] = ctrl
        step_sim()

    cube_final_z = data.qpos[cube_qpos_addr + 2]
    grasping = is_grasping()
    lifted = cube_final_z > 0.05

    print(f"\n=== Result ===")
    print(f"Cube Z: {cube_final_z:.4f}")
    print(f"Still grasping: {grasping}")
    print(f"Lifted: {lifted}")

    return grasping, lifted, cube_final_z


if __name__ == "__main__":
    use_viewer = "--viewer" in sys.argv

    # Test configurations
    configs = [
        # (wrist_flex, wrist_roll, approach, description)
        (1.65, np.pi/2, "top", "Top-down (current working approach)"),
        (0.0, 0.0, "side", "Pure horizontal side approach"),
        (0.8, np.pi/2, "side", "Angled side approach"),
    ]

    if use_viewer:
        # Interactive: run first config
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wrist_flex, wrist_roll, approach, desc = configs[0]
            print(f"\nTesting: {desc}")
            run_grasp(wrist_flex, wrist_roll, approach, viewer)
            print("\nPress Ctrl+C to exit...")
            import time
            while viewer.is_running():
                time.sleep(0.1)
    else:
        # Run all configs
        results = []
        for wrist_flex, wrist_roll, approach, desc in configs:
            print(f"\n{'='*60}")
            print(f"Testing: {desc}")
            grasped, lifted, cube_z = run_grasp(wrist_flex, wrist_roll, approach)
            results.append((desc, lifted))

        print(f"\n{'='*60}")
        print("SUMMARY:")
        for desc, lifted in results:
            status = "SUCCESS" if lifted else "FAIL"
            print(f"  [{status}] {desc}")
