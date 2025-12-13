"""Pick and place GoalEnv for HER (Hindsight Experience Replay)."""
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class PickCubeGoalEnv(gym.Env):
    """GoalEnv for picking a cube and placing it at a target location.

    This environment follows the GoalEnv API required for HER:
    - Dictionary observation space with 'observation', 'achieved_goal', 'desired_goal'
    - Vectorized compute_reward() method

    Observation space (Dict):
        - observation: joint pos (6) + vel (6) + gripper pos (3) + cube pos (3) = 18
        - achieved_goal: cube position (3)
        - desired_goal: target position (3)

    Action space (6 dims):
        - Delta joint positions for all 6 joints (continuous)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        action_scale: float = 0.1,
        reward_type: str = "sparse",  # "sparse" or "dense"
        distance_threshold: float = 0.03,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self._step_count = 0

        # Load model from SO-ARM100 directory
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

        # GoalEnv observation space (Dict)
        obs_dim = 6 + 6 + 3 + 3  # joint pos + vel + gripper pos + cube pos
        goal_dim = 3  # xyz position

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            "achieved_goal": spaces.Box(
                low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
            ),
        })

        # Fixed target position (inside bowl)
        self.target_pos = np.array([0.40, 0.10, 0.04], dtype=np.float32)

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Get current observation in GoalEnv format."""
        # Joint positions and velocities
        joint_pos = self.data.qpos[: self.n_joints].copy()
        joint_vel = self.data.qvel[: self.n_joints].copy()

        # Gripper position (from sensor)
        gripper_pos = self.data.sensor("gripper_pos").data.copy()

        # Cube position (from sensor) - this is the achieved_goal
        cube_pos = self.data.sensor("cube_pos").data.copy()

        observation = np.concatenate([
            joint_pos, joint_vel, gripper_pos, cube_pos
        ]).astype(np.float32)

        return {
            "observation": observation,
            "achieved_goal": cube_pos.astype(np.float32),
            "desired_goal": self.target_pos.copy(),
        }

    def _get_info(self) -> dict[str, Any]:
        """Get additional info."""
        cube_pos = self.data.sensor("cube_pos").data.copy()
        cube_to_target = np.linalg.norm(cube_pos - self.target_pos)

        return {
            "is_success": cube_to_target < self.distance_threshold,
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Compute reward for HER.

        This method must be vectorized to support batched goal relabeling.

        Args:
            achieved_goal: (batch, 3) or (3,) - cube positions
            desired_goal: (batch, 3) or (3,) - target positions
            info: Additional info (not used for sparse/dense)

        Returns:
            Reward array matching batch dimension
        """
        # Compute distances (vectorized)
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if self.reward_type == "sparse":
            # -1 if not at goal, 0 if at goal
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            # Dense: negative distance
            return -d.astype(np.float32)

    def compute_terminated(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> bool:
        """Check if episode should terminate (goal reached)."""
        d = np.linalg.norm(achieved_goal - desired_goal)
        return bool(d < self.distance_threshold)

    def compute_truncated(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> bool:
        """Check if episode should be truncated (max steps)."""
        return self._step_count >= self.max_episode_steps

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Randomize cube position slightly (within arm's reach)
        if self.np_random is not None:
            cube_x = 0.40 + self.np_random.uniform(-0.03, 0.03)
            cube_y = -0.10 + self.np_random.uniform(-0.03, 0.03)
            # Set cube position (freejoint: 3 pos + 4 quat)
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
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
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

        # Compute reward using the GoalEnv method
        reward = float(self.compute_reward(
            obs["achieved_goal"], obs["desired_goal"], info
        ))

        # Check termination/truncation
        terminated = self.compute_terminated(
            obs["achieved_goal"], obs["desired_goal"], info
        )
        truncated = self.compute_truncated(
            obs["achieved_goal"], obs["desired_goal"], info
        )

        return obs, reward, terminated, truncated, info

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
