"""Pick and place environment for SO-101 arm."""
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class PickCubeEnv(gym.Env):
    """Environment for picking a cube and placing it at a target location.

    Observation space (21 dims):
        - Joint positions (6)
        - Joint velocities (6)
        - Gripper position (3)
        - Cube position (3)
        - Target position (3) - note: this is fixed but included for generalization

    Action space (6 dims):
        - Delta joint positions for all 6 joints (continuous)

    Staged Rewards:
        1. Reach: gripper approaching cube
        2. Grasp: bonus when gripper closes around cube
        3. Lift: bonus when cube is lifted off ground while grasping
        4. Place: cube approaching target (only when lifted)
        5. Success: cube at target
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        action_scale: float = 0.1,
        # Reward config
        reward_config: dict | None = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self._step_count = 0

        # Reward configuration with defaults
        reward_config = reward_config or {}
        self.grasp_distance_threshold = reward_config.get("grasp_distance_threshold", 0.03)
        self.gripper_closed_threshold = reward_config.get("gripper_closed_threshold", 0.3)
        self.lift_height_threshold = reward_config.get("lift_height_threshold", 0.03)
        self.success_threshold = reward_config.get("success_threshold", 0.03)

        self.reach_weight = reward_config.get("reach_weight", 1.0)
        self.grasp_bonus = reward_config.get("grasp_bonus", 1.0)
        self.lift_bonus = reward_config.get("lift_bonus", 2.0)
        self.lift_height_scale = reward_config.get("lift_height_scale", 10.0)
        self.place_weight = reward_config.get("place_weight", 2.0)
        self.success_bonus = reward_config.get("success_bonus", 10.0)
        self.drop_penalty = reward_config.get("drop_penalty", 5.0)

        # Load model from SO-ARM100 directory where meshes are located
        scene_path = Path(__file__).parent.parent / "SO-ARM100/Simulation/SO101/pick_cube_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Get joint and actuator info
        self.n_joints = 6
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        # Control ranges from actuators
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Action space: delta joint positions (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Observation space
        obs_dim = 6 + 6 + 3 + 3 + 3  # joints pos + vel + gripper pos + cube pos + target pos
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Target position: inside bowl (bowl is at y=0.10, target z=0.04 above ground)
        self.target_pos = np.array([0.40, 0.10, 0.04])

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Joint positions and velocities
        joint_pos = self.data.qpos[: self.n_joints].copy()
        joint_vel = self.data.qvel[: self.n_joints].copy()

        # Gripper position (from sensor)
        gripper_pos = self.data.sensor("gripper_pos").data.copy()

        # Cube position (from sensor)
        cube_pos = self.data.sensor("cube_pos").data.copy()

        return np.concatenate(
            [joint_pos, joint_vel, gripper_pos, cube_pos, self.target_pos]
        ).astype(np.float32)

    def _get_gripper_state(self) -> float:
        """Get gripper joint position (lower = more closed)."""
        gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
        gripper_qpos_addr = self.model.jnt_qposadr[gripper_joint_id]
        return self.data.qpos[gripper_qpos_addr]

    def _has_cube_contact(self) -> bool:
        """Check if gripper is in contact with the cube."""
        cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        # Gripper geoms are 25-30 (gripper body and moving_jaw)
        gripper_geom_ids = set(range(25, 31))

        for i in range(self.data.ncon):
            geom1 = self.data.contact[i].geom1
            geom2 = self.data.contact[i].geom2

            # Check if contact is between gripper and cube
            if geom1 == cube_geom_id and geom2 in gripper_geom_ids:
                return True
            if geom2 == cube_geom_id and geom1 in gripper_geom_ids:
                return True

        return False

    def _is_grasping(self, gripper_pos: np.ndarray, cube_pos: np.ndarray) -> bool:
        """Check if gripper is grasping the cube.

        Requires BOTH:
        1. Physical contact between gripper and cube
        2. Gripper is closed
        """
        gripper_state = self._get_gripper_state()
        is_closed = gripper_state < self.gripper_closed_threshold
        has_contact = self._has_cube_contact()

        return is_closed and has_contact

    def _get_info(self) -> dict[str, Any]:
        """Get additional info."""
        gripper_pos = self.data.sensor("gripper_pos").data.copy()
        cube_pos = self.data.sensor("cube_pos").data.copy()

        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_to_target = np.linalg.norm(cube_pos - self.target_pos)
        cube_z = cube_pos[2]

        has_contact = self._has_cube_contact()
        is_grasping = self._is_grasping(gripper_pos, cube_pos)
        is_lifted = is_grasping and cube_z > self.lift_height_threshold

        return {
            "gripper_to_cube": gripper_to_cube,
            "cube_to_target": cube_to_target,
            "cube_pos": cube_pos.copy(),
            "gripper_pos": gripper_pos.copy(),
            "gripper_state": self._get_gripper_state(),
            "has_contact": has_contact,
            "is_grasping": is_grasping,
            "is_lifted": is_lifted,
            "is_success": cube_to_target < self.success_threshold,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Randomize cube position slightly (within arm's reach)
        if self.np_random is not None:
            cube_x = 0.40 + self.np_random.uniform(-0.03, 0.03)
            cube_y = -0.10 + self.np_random.uniform(-0.03, 0.03)
            # Set cube position (freejoint: 3 pos + 4 quat)
            # Cube is 2cm (0.01 half-size), so z=0.01 puts it on ground
            cube_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
            )
            cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
            self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, 0.01]
            self.data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

        # Step once to update sensors
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step."""
        # Scale action and apply as delta to current control
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.action_scale

        # Get current joint positions and add delta
        current_ctrl = self.data.ctrl.copy()
        new_ctrl = current_ctrl + delta

        # Clip to control ranges
        new_ctrl = np.clip(new_ctrl, self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1])
        self.data.ctrl[:] = new_ctrl

        # Step simulation (multiple physics steps per control step)
        n_substeps = 10
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()

        # Calculate reward
        reward = self._compute_reward(info)

        # Check termination
        terminated = info["is_success"]
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, info: dict[str, Any]) -> float:
        """Compute staged reward based on current state.

        Stages:
        1. Reach: reward for gripper approaching cube
        2. Grasp: bonus when gripper closes around cube
        3. Lift: bonus when cube is lifted while grasping
        4. Place: reward for moving lifted cube toward target
        5. Success: large bonus when cube at target
        """
        gripper_to_cube = info["gripper_to_cube"]
        cube_to_target = info["cube_to_target"]
        cube_z = info["cube_pos"][2]
        is_grasping = info["is_grasping"]
        is_lifted = info["is_lifted"]

        reward = 0.0

        # Stage 1: Reach - always encourage gripper to approach cube
        reach_reward = -self.reach_weight * gripper_to_cube
        reward += reach_reward

        # Stage 2: Grasp bonus - reward for closing gripper around cube
        if is_grasping:
            reward += self.grasp_bonus

        # Stage 3: Lift bonus - reward for lifting cube off ground
        if is_lifted:
            lift_reward = self.lift_bonus + cube_z * self.lift_height_scale
            reward += lift_reward

            # Stage 4: Place - only encourage moving to target when lifted
            place_reward = -self.place_weight * cube_to_target
            reward += place_reward

        # Stage 5: Success bonus
        if info["is_success"]:
            reward += self.success_bonus

        # Penalty if cube falls off table
        if cube_z < 0.0:
            reward -= self.drop_penalty

        return reward

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
