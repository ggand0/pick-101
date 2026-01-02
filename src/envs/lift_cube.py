"""Lift cube environment with Cartesian (end-effector) action space.

Uses IK controller internally - agent outputs delta XYZ + gripper,
which is much easier to explore than joint velocities.
"""
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.controllers.ik_controller import IKController


class LiftCubeCartesianEnv(gym.Env):
    """Lift cube with Cartesian action space.

    Action space (4 dims):
        - Delta X, Y, Z for end-effector position
        - Gripper open/close (-1 to 1)

    Observation space (21 dims):
        - Joint positions (6)
        - Joint velocities (6)
        - Gripper position (3)
        - Gripper orientation (3) - euler angles
        - Cube position (3)

    This is much easier to learn than joint-space control because:
    - Random actions naturally explore 3D space
    - Agent doesn't need to learn arm kinematics
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        action_scale: float = 0.02,  # 2cm per step max
        lift_height: float = 0.08,
        hold_steps: int = 10,
        reward_type: str = "dense",
        reward_version: str = "v7",
        curriculum_stage: int = 0,  # 0=normal, 1=cube in gripper lifted, 2=cube in gripper on table, 3=gripper near cube
        lock_wrist: bool = False,  # Lock wrist joints for stable grasping
        place_target: tuple[float, float] | None = None,  # (x, y) target for placement
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.lift_height = lift_height
        self.hold_steps = hold_steps
        self.reward_type = reward_type
        self.reward_version = reward_version
        self.curriculum_stage = curriculum_stage
        self.lock_wrist = lock_wrist
        self.place_target = place_target
        self._step_count = 0
        self._hold_count = 0
        self._was_grasping = False  # Track if we had grasp in previous step (for drop penalty)
        self._reset_gripper_action = None  # Gripper action used at reset (for curriculum)
        self._prev_action = np.zeros(4)  # Track previous action for smoothness penalty
        self._place_target_pos = None  # Full 3D target position (set at reset)

        # Load model
        scene_path = Path(__file__).parent.parent.parent / "models/so101/lift_cube.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Get finger pad geom IDs for contact detection (new model with finger pads)
        self._static_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        self._moving_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        # Initialize IK controller with gripperframe (TCP)
        # For grasping, we use offset compensation: target = finger_target + (TCP - finger_mid)
        self.ik = IKController(self.model, self.data, end_effector_site="gripperframe")

        # Joint info
        self.n_joints = 6
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Action space: delta XYZ + gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space
        # Base: joints pos/vel + gripper pos + gripper euler + cube pos
        # With place_target: add target pos (3)
        obs_dim = 6 + 6 + 3 + 3 + 3
        if self.place_target is not None:
            obs_dim += 3  # Add target position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Track target EE position
        self._target_ee_pos = None

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[: self.n_joints].copy()
        joint_vel = self.data.qvel[: self.n_joints].copy()
        gripper_pos = self.ik.get_ee_position()

        # Get gripper orientation as euler angles
        gripper_mat = self.ik.get_ee_orientation()
        gripper_euler = self._rotation_matrix_to_euler(gripper_mat)

        cube_pos = self.data.sensor("cube_pos").data.copy()

        obs_parts = [joint_pos, joint_vel, gripper_pos, gripper_euler, cube_pos]

        # Include target position if place task
        if self._place_target_pos is not None:
            obs_parts.append(self._place_target_pos)

        return np.concatenate(obs_parts).astype(np.float32)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.array([roll, pitch, yaw])

    def _get_gripper_state(self) -> float:
        gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
        gripper_qpos_addr = self.model.jnt_qposadr[gripper_joint_id]
        return self.data.qpos[gripper_qpos_addr]

    def _check_cube_contacts(self) -> tuple[bool, bool]:
        """Check if cube contacts static finger pad and moving finger pad.

        Uses named geom lookups for the finger pad collision boxes.
        """
        has_static_contact = False
        has_moving_contact = False

        for i in range(self.data.ncon):
            geom1 = self.data.contact[i].geom1
            geom2 = self.data.contact[i].geom2

            other_geom = None
            if geom1 == self._cube_geom_id:
                other_geom = geom2
            elif geom2 == self._cube_geom_id:
                other_geom = geom1

            if other_geom is not None:
                if other_geom == self._static_pad_geom_id:
                    has_static_contact = True
                if other_geom == self._moving_pad_geom_id:
                    has_moving_contact = True

        return has_static_contact, has_moving_contact

    def _is_grasping(self) -> bool:
        """Check if cube is properly grasped (pinched between gripper and jaw)."""
        gripper_state = self._get_gripper_state()
        is_closed = gripper_state < 0.25

        has_gripper_contact, has_jaw_contact = self._check_cube_contacts()
        is_pinched = has_gripper_contact and has_jaw_contact

        return is_closed and is_pinched

    def _get_info(self) -> dict[str, Any]:
        gripper_pos = self.ik.get_ee_position()
        cube_pos = self.data.sensor("cube_pos").data.copy()

        # Use gripper_pos (TCP at fingertips) for reach reward
        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_z = cube_pos[2]
        is_grasping = self._is_grasping()
        is_lifted = is_grasping and cube_z > self.lift_height

        has_gripper_contact, has_jaw_contact = self._check_cube_contacts()

        info = {
            "gripper_to_cube": gripper_to_cube,
            "cube_pos": cube_pos.copy(),
            "cube_z": cube_z,
            "gripper_pos": gripper_pos.copy(),
            "gripper_state": self._get_gripper_state(),
            "has_gripper_contact": has_gripper_contact,
            "has_jaw_contact": has_jaw_contact,
            "is_grasping": is_grasping,
            "is_lifted": is_lifted,
            "hold_count": self._hold_count,
            "is_success": self._hold_count >= self.hold_steps,
        }

        # Add place-related info if place target is set
        if self._place_target_pos is not None:
            cube_to_target_xy = np.linalg.norm(cube_pos[:2] - self._place_target_pos[:2])
            cube_at_target = cube_to_target_xy < 0.02  # Within 2cm of target
            cube_on_ground = cube_z < 0.025  # Cube resting on table
            is_placed = cube_at_target and cube_on_ground and not is_grasping
            info["cube_to_target"] = cube_to_target_xy
            info["place_target"] = self._place_target_pos.copy()
            info["is_placed"] = is_placed
            # Override success for place task
            info["is_success"] = is_placed

        return info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Get cube joint info
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        if self.curriculum_stage == 0:
            # Stage 0: Normal - cube on table, arm at home position
            if self.np_random is not None:
                cube_x = 0.40 + self.np_random.uniform(-0.03, 0.03)
                cube_y = -0.10 + self.np_random.uniform(-0.03, 0.03)
                self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, 0.01]
                self.data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

        elif self.curriculum_stage == 1:
            # Stage 1: Cube in closed gripper at lift height (easiest - just hold)
            self._reset_with_cube_in_gripper(cube_qpos_addr, lift_height=self.lift_height)

        elif self.curriculum_stage == 2:
            # Stage 2: Cube in closed gripper at table level (need to lift)
            self._reset_with_cube_in_gripper(cube_qpos_addr, lift_height=0.03)

        elif self.curriculum_stage == 3:
            # Stage 3: Gripper near cube, open (need to close and lift)
            self._reset_gripper_near_cube(cube_qpos_addr)

        elif self.curriculum_stage == 4:
            # Stage 4: Gripper far from cube (need to reach, descend, grasp, lift)
            self._reset_gripper_far_from_cube(cube_qpos_addr)

        mujoco.mj_forward(self.model, self.data)

        # Initialize target EE position to current position
        self._target_ee_pos = self.ik.get_ee_position().copy()

        # Initialize place target if set
        if self.place_target is not None:
            # Add small randomization to place target
            if self.np_random is not None:
                target_x = self.place_target[0] + self.np_random.uniform(-0.01, 0.01)
                target_y = self.place_target[1] + self.np_random.uniform(-0.01, 0.01)
            else:
                target_x, target_y = self.place_target
            self._place_target_pos = np.array([target_x, target_y, 0.015])  # z = cube resting height
        else:
            self._place_target_pos = None

        self._step_count = 0
        self._hold_count = 0
        self._was_grasping = False
        self._prev_action = np.zeros(4)

        return self._get_obs(), self._get_info()

    def _get_finger_mid(self) -> np.ndarray:
        """Get midpoint between gripper fingers."""
        f28 = self.data.geom_xpos[28]
        f30 = self.data.geom_xpos[30]
        return (f28 + f30) / 2

    def _get_cube_contacts(self, cube_geom_id: int) -> list[int]:
        """Get list of geom IDs in contact with cube."""
        contacts = []
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact[i].geom1, self.data.contact[i].geom2
            if g1 == cube_geom_id or g2 == cube_geom_id:
                other = g2 if g1 == cube_geom_id else g1
                contacts.append(other)
        return contacts

    def _reset_with_cube_in_gripper(self, cube_qpos_addr: int, lift_height: float):
        """Reset with cube grasped in gripper at specified height.

        Exact logic from test_topdown_pick.py pick_up_block().
        """
        # Constants from test_topdown_pick.py
        height_offset = 0.03
        gripper_open = 0.3  # partially open (ECE4560 uses 50/100, not fully open)
        gripper_closed = -0.8
        grasp_z_offset = 0.005
        finger_width_offset = -0.015
        locked_joints = [3, 4]

        # Randomize cube position slightly
        if self.np_random is not None:
            cube_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            cube_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            cube_x, cube_y = 0.25, 0.0
        cube_z = 0.015

        # Place cube on table
        self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

        # Top-down arm configuration
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)

        # Let cube settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Read actual cube position after settle
        actual_cube_pos = self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy()

        # Step 1: Move above block with gripper open
        above_pos = actual_cube_pos.copy()
        above_pos[2] += grasp_z_offset + height_offset
        above_pos[1] += finger_width_offset
        for _ in range(300):
            ctrl = self.ik.step_toward_target(above_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        # Step 2: Move down to block with gripper open
        grasp_target = actual_cube_pos.copy()
        grasp_target[2] += grasp_z_offset
        grasp_target[1] += finger_width_offset
        for _ in range(200):
            ctrl = self.ik.step_toward_target(grasp_target, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        # Step 3: Close gripper with contact detection
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

            ctrl = self.ik.step_toward_target(grasp_target, gripper_action=gripper, gain=0.5, locked_joints=locked_joints)
            ctrl[3] = np.pi / 2
            ctrl[4] = np.pi / 2
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

            # Detect contact using is_grasping logic
            if self._is_grasping() and contact_step is None:
                contact_step = step
                contact_action = gripper

            # Check if done tightening
            if contact_step is not None:
                target_action = max(contact_action - tighten_amount, -1.0)
                if gripper <= target_action + 0.01:
                    grasp_action = gripper
                    break

        # Handle case where loop finished without breaking
        if contact_step is not None and gripper > max(contact_action - tighten_amount, -1.0) + 0.01:
            grasp_action = gripper

        # Store the gripper action for curriculum learning
        self._reset_gripper_action = grasp_action

        # Agent starts here with cube grasped - needs to learn to lift

    def _reset_gripper_near_cube(self, cube_qpos_addr: int):
        """Reset with gripper positioned above cube, open, ready to grasp.

        Uses same top-down setup as stage 1/2 but with gripper open and
        positioned slightly above grasp height.
        """
        # Constants from test_topdown_pick.py
        height_offset = 0.03
        gripper_open = 0.3  # partially open
        grasp_z_offset = 0.005
        finger_width_offset = -0.015

        # Randomize cube position (same range as stage 1)
        if self.np_random is not None:
            cube_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            cube_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            cube_x, cube_y = 0.25, 0.0
        cube_z = 0.015

        # Place cube on table
        self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

        # Top-down arm configuration (wrist locked)
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)

        # Let cube settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Read actual cube position after settle
        actual_cube_pos = self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy()

        # Position gripper above cube (above grasp height, so agent needs to lower)
        above_pos = actual_cube_pos.copy()
        above_pos[1] += finger_width_offset
        above_pos[2] = actual_cube_pos[2] + grasp_z_offset + height_offset  # Above grasp height

        # Move to above position with gripper open
        locked_joints = [3, 4] if self.lock_wrist else []
        for _ in range(100):
            ctrl = self.ik.step_toward_target(
                above_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints
            )
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def _reset_gripper_far_from_cube(self, cube_qpos_addr: int):
        """Reset with gripper positioned further from cube - must reach first.

        Stage 4: Gripper starts ~8-12cm away from cube in XY plane,
        at lift height. Agent must reach, descend, grasp, and lift.
        """
        gripper_open = 0.3

        # Randomize cube position
        if self.np_random is not None:
            cube_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            cube_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            cube_x, cube_y = 0.25, 0.0
        cube_z = 0.015

        # Place cube on table
        self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

        # Top-down arm configuration (wrist locked)
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)

        # Let cube settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Read actual cube position after settle
        actual_cube_pos = self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy()

        # Position gripper at a distance from cube
        # Random offset in XY: 8-12cm away, random direction
        if self.np_random is not None:
            angle = self.np_random.uniform(0, 2 * np.pi)
            distance = self.np_random.uniform(0.08, 0.12)
            offset_x = distance * np.cos(angle)
            offset_y = distance * np.sin(angle)
        else:
            offset_x, offset_y = 0.10, 0.0

        start_pos = np.array([
            actual_cube_pos[0] + offset_x,
            actual_cube_pos[1] + offset_y,
            self.lift_height + 0.02  # Start at lift height
        ])

        # Clamp to workspace bounds
        start_pos[0] = np.clip(start_pos[0], 0.15, 0.45)
        start_pos[1] = np.clip(start_pos[1], -0.25, 0.25)

        # Move to start position with gripper open
        locked_joints = [3, 4] if self.lock_wrist else []
        for _ in range(150):
            ctrl = self.ik.step_toward_target(
                start_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints
            )
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Parse action
        action = np.clip(action, -1.0, 1.0)
        delta_xyz = action[:3] * self.action_scale
        gripper_action = action[3]

        # Update target end-effector position
        self._target_ee_pos += delta_xyz

        # Clamp to workspace bounds
        self._target_ee_pos[0] = np.clip(self._target_ee_pos[0], 0.1, 0.5)
        self._target_ee_pos[1] = np.clip(self._target_ee_pos[1], -0.3, 0.3)
        self._target_ee_pos[2] = np.clip(self._target_ee_pos[2], 0.01, 0.4)

        # Use IK to compute joint controls
        if self.place_target is not None:
            # Place task: agent controls gripper (to release), but wrist stays locked
            ctrl = self.ik.step_toward_target(
                self._target_ee_pos,
                gripper_action=gripper_action,
                gain=0.5,
                locked_joints=[4],
            )
            ctrl[4] = np.pi / 2  # wrist_roll horizontal
        elif self.lock_wrist:
            # Lift task: lock gripper to reset value - agent only controls XYZ
            # This prevents physics issues (can't open with cube in grip, closing launches cube)
            if self._reset_gripper_action is not None:
                stable_gripper = self._reset_gripper_action
            else:
                stable_gripper = gripper_action

            # Lock only wrist_roll (joint 4) for horizontal fingers
            # Let wrist_flex (joint 3) move for lifting range
            ctrl = self.ik.step_toward_target(
                self._target_ee_pos,
                gripper_action=stable_gripper,
                gain=0.5,
                locked_joints=[4],
            )
            ctrl[4] = np.pi / 2  # wrist_roll horizontal
        else:
            ctrl = self.ik.step_toward_target(
                self._target_ee_pos,
                gripper_action=gripper_action,
                gain=0.5,
            )

        # Apply control and step simulation multiple times for stability
        self.data.ctrl[:] = ctrl
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get state
        obs = self._get_obs()
        info = self._get_info()

        # Update hold counter (for lift task)
        if info["is_lifted"]:
            self._hold_count += 1
        else:
            self._hold_count = 0
        info["hold_count"] = self._hold_count

        # Check success (depends on task type)
        if self._place_target_pos is not None:
            # Place task: success is determined by is_placed (set in _get_info)
            is_success = info.get("is_placed", False)
        else:
            # Lift task: success after holding for hold_steps
            is_success = self._hold_count >= self.hold_steps
        info["is_success"] = is_success

        # Compute reward (pass previous grasp state for drop penalty)
        reward = self._compute_reward(info, was_grasping=self._was_grasping, action=action)

        # Update previous grasp state and action for next step
        self._was_grasping = info["is_grasping"]
        self._prev_action = action.copy()

        terminated = is_success
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        if self.reward_type == "sparse":
            return 0.0 if info["is_success"] else -1.0

        # Dispatch to version-specific reward function
        reward_fn = getattr(self, f"_reward_{self.reward_version}", None)
        if reward_fn is None:
            raise ValueError(f"Unknown reward version: {self.reward_version}")
        return reward_fn(info, was_grasping=was_grasping, action=action)

    def _reward_v1(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V1: Reach + grasp bonus + binary lift. Original reward that achieved grasping
        (via physics exploit with soft contacts)."""
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Grasp bonus (always)
        if info["is_grasping"]:
            reward += 0.25

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v2(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V2: Reach + continuous lift (no grasp condition). Disrupted grasping entirely."""
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Grasp bonus (stronger than V1)
        if info["is_grasping"]:
            reward += 0.5

        # Continuous lift (unconditional - this is what broke it)
        reward += max(0, (cube_z - 0.01) * 50.0)

        # Target height bonus
        if cube_z > self.lift_height:
            reward += 2.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v3(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V3: V1 + continuous lift gradient. Destabilized training."""
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Continuous lift baseline (without grasp)
        lift_baseline = max(0, (cube_z - 0.01) * 10.0)
        reward += lift_baseline

        # Grasp bonus (always)
        if info["is_grasping"]:
            reward += 0.25
            # Stronger lift reward when grasping
            reward += (cube_z - 0.01) * 40.0

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v4(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V4: V3 but grasp bonus only when elevated. Never closes gripper."""
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Continuous lift baseline
        lift_baseline = max(0, (cube_z - 0.01) * 10.0)
        reward += lift_baseline

        # Grasp bonus only when elevated
        if info["is_grasping"] and cube_z > 0.01:
            reward += 0.25
            reward += (cube_z - 0.01) * 40.0

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v5(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V5: V3 + push-down penalty. Nudge exploit - tilts cube."""
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Continuous lift baseline
        lift_baseline = max(0, (cube_z - 0.01) * 10.0)
        reward += lift_baseline

        # Grasp bonus (always)
        if info["is_grasping"]:
            reward += 0.25
            reward += (cube_z - 0.01) * 40.0

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v6(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V6: V5 without lift_baseline. Safe hover far away."""
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Grasp bonus (always)
        if info["is_grasping"]:
            reward += 0.25
            reward += (cube_z - 0.01) * 40.0

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v7(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V7: V1 + push-down penalty. Prevents agent from pushing cube into table."""
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Grasp bonus (always)
        if info["is_grasping"]:
            reward += 0.25

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v8(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V8: V7 + drop penalty. Penalizes losing grasp after having it.

        For curriculum learning where agent starts with cube grasped,
        this prevents the agent from learning to drop the cube.
        """
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]
        is_grasping = info["is_grasping"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Drop penalty: penalize losing grasp after having it
        if was_grasping and not is_grasping:
            reward -= 2.0  # Significant penalty for dropping

        # Grasp bonus (always)
        if is_grasping:
            reward += 0.25

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v9(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V9: V8 + continuous lift gradient. For curriculum learning.

        Adds smooth reward for lifting higher while grasping, instead of
        just binary threshold at z=0.02.
        """
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]
        is_grasping = info["is_grasping"]

        # Reach reward (less important for curriculum where we start grasped)
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Drop penalty: penalize losing grasp after having it
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus
        if is_grasping:
            reward += 0.25

            # Continuous lift reward when grasping - this is the key addition
            # Reward proportional to height above table (0.015 is cube resting height)
            lift_progress = max(0, cube_z - 0.015) / (self.lift_height - 0.015)
            reward += lift_progress * 2.0  # Up to +2.0 at target height

        # Binary lift bonus (kept for compatibility)
        if cube_z > 0.02:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v10(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V10: v9 + target height bonus.

        Only change from v9: +1.0 bonus when z is within 5mm of target.
        """
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]
        is_grasping = info["is_grasping"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus
        if is_grasping:
            reward += 0.25

            # Continuous lift reward when grasping
            lift_progress = max(0, cube_z - 0.015) / (self.lift_height - 0.015)
            reward += lift_progress * 2.0

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Target height bonus (the only addition from v9)
        if abs(cube_z - self.lift_height) < 0.005:
            reward += 1.0

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v11(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V11: v10 + action rate penalty for smooth control.

        Penalizes rapid action changes to reduce twitching/oscillation.
        """
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]
        is_grasping = info["is_grasping"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus
        if is_grasping:
            reward += 0.25

            # Continuous lift reward when grasping
            lift_progress = max(0, cube_z - 0.015) / (self.lift_height - 0.015)
            reward += lift_progress * 2.0

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Target height bonus (aligned with success: z > lift_height)
        if cube_z > self.lift_height:
            reward += 1.0

        # Action rate penalty for smoothness (only when lifted, to not hinder lifting)
        if action is not None and cube_z > 0.06:
            action_delta = action - self._prev_action
            action_penalty = 0.01 * np.sum(action_delta**2)
            reward -= action_penalty

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v12(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V12: Pick-and-place reward. Extends v11 with transport and placement rewards.

        Phases:
        1. Reach and grasp (same as v11)
        2. Lift (same as v11)
        3. Transport to target (reward for moving cube toward target)
        4. Lower and release (reward for placing at target)
        """
        reward = 0.0
        cube_pos = info["cube_pos"]
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]
        is_grasping = info["is_grasping"]

        # Phase 1: Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Drop penalty (only during transport, not during intentional release)
        cube_to_target = info.get("cube_to_target", 0)
        if was_grasping and not is_grasping and cube_to_target > 0.03:
            reward -= 2.0  # Penalty for dropping away from target

        # Phase 2: Grasp and lift rewards
        if is_grasping:
            reward += 0.25

            # Continuous lift reward when grasping
            lift_progress = max(0, cube_z - 0.015) / (self.lift_height - 0.015)
            reward += lift_progress * 2.0

        # Binary lift bonus
        if cube_z > 0.02:
            reward += 1.0

        # Target height bonus
        if cube_z > self.lift_height:
            reward += 1.0

        # Phase 3: Transport reward (move cube toward target while lifted)
        if self._place_target_pos is not None:
            target_xy = self._place_target_pos[:2]
            cube_xy = cube_pos[:2]

            # Reward for cube being close to target (XY only)
            transport_reward = 1.0 - np.tanh(5.0 * cube_to_target)
            reward += transport_reward

            # Bonus for reaching target zone while grasping and lifted
            if cube_to_target < 0.03 and is_grasping and cube_z > self.lift_height:
                reward += 2.0  # At target, ready to place

            # Phase 4: Placement reward
            if cube_to_target < 0.03:
                # Reward for lowering cube at target
                if cube_z < self.lift_height:
                    lower_progress = (self.lift_height - cube_z) / (self.lift_height - 0.015)
                    reward += lower_progress * 1.0

                # Reward for releasing at target (gripper opening)
                if not is_grasping and cube_z < 0.025:
                    reward += 3.0  # Just released at target

        # Action rate penalty for smoothness (only when lifted)
        if action is not None and cube_z > 0.06:
            action_delta = action - self._prev_action
            action_penalty = 0.01 * np.sum(action_delta**2)
            reward -= action_penalty

        # Success bonus
        if info.get("is_placed", False):
            reward += 10.0

        return reward

    def _reward_v13(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V13: v11 with binary lift bonus gated on is_grasping.

        Fixes exploit where agent tilts cube to get lift bonus without proper grasp.
        """
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]
        is_grasping = info["is_grasping"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus
        if is_grasping:
            reward += 0.25

            # Continuous lift reward when grasping
            lift_progress = max(0, cube_z - 0.015) / (self.lift_height - 0.015)
            reward += lift_progress * 2.0

            # Binary lift bonus (NOW GATED on is_grasping)
            if cube_z > 0.02:
                reward += 1.0

        # Target height bonus (aligned with success: z > lift_height)
        if cube_z > self.lift_height:
            reward += 1.0

        # Action rate penalty for smoothness (only when lifted, to not hinder lifting)
        if action is not None and cube_z > 0.06:
            action_delta = action - self._prev_action
            action_penalty = 0.01 * np.sum(action_delta**2)
            reward -= action_penalty

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v14(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V14: v13 with action penalty only during hold phase.

        Fixes the 7cm plateau issue where action penalty at 6cm+ blocked the final push to 8cm.
        Now action penalty only applies during hold phase (hold_count > 0), allowing free
        movement during lifting while encouraging stability once at target height.
        """
        reward = 0.0
        cube_z = info["cube_z"]
        gripper_to_cube = info["gripper_to_cube"]
        is_grasping = info["is_grasping"]
        hold_count = info["hold_count"]

        # Reach reward
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
        reward += reach_reward

        # Push-down penalty
        if cube_z < 0.01:
            push_penalty = (0.01 - cube_z) * 50.0
            reward -= push_penalty

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus
        if is_grasping:
            reward += 0.25

            # Continuous lift reward when grasping
            lift_progress = max(0, cube_z - 0.015) / (self.lift_height - 0.015)
            reward += lift_progress * 2.0

            # Binary lift bonus (gated on is_grasping)
            if cube_z > 0.02:
                reward += 1.0

        # Target height bonus (aligned with success: z > lift_height)
        if cube_z > self.lift_height:
            reward += 1.0

        # Action rate penalty ONLY during hold phase (not before reaching target)
        # This allows free movement during lifting, but encourages stability once at target
        if action is not None and hold_count > 0:
            action_delta = action - self._prev_action
            action_penalty = 0.02 * np.sum(action_delta**2)
            reward -= action_penalty

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def render(self, camera: str = "closeup") -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            cam = mujoco.MjvCamera()
            if camera == "closeup":
                # Side view close to cube
                cam.lookat[:] = [0.40, -0.10, 0.03]
                cam.distance = 0.35
                cam.azimuth = 90
                cam.elevation = -15
            elif camera == "wide":
                # Diagonal view of arm and cube
                cam.lookat[:] = [0.25, -0.05, 0.05]
                cam.distance = 0.8
                cam.azimuth = 135
                cam.elevation = -25
            else:  # "wide2"
                # Diagonal view from other side
                cam.lookat[:] = [0.25, -0.05, 0.05]
                cam.distance = 0.8
                cam.azimuth = 45
                cam.elevation = -25
            self._renderer.update_scene(self.data, camera=cam)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
