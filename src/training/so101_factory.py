"""SO-101 environment factory for RoboBase."""

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig

from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    ActionSequence,
    AppendDemoInfo,
    ConcatDim,
    FrameStack,
    OnehotTime,
    RescaleFromTanh,
)

from src.envs.lift_cube import LiftCubeCartesianEnv


class SuccessInfoWrapper(gym.Wrapper):
    """Wrapper that copies is_success to task_success for RoboBase compatibility."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # RoboBase looks for 'task_success', our env returns 'is_success'
        if "is_success" in info:
            info["task_success"] = float(info["is_success"])
        return obs, reward, terminated, truncated, info


class WristCameraWrapper(gym.ObservationWrapper):
    """Adds wrist camera image to observation dict for RoboBase.

    Includes proprioceptive state (joints, gripper) but excludes cube_pos
    which must be inferred from vision for sim-to-real transfer.

    Matches real camera preprocessing:
    1. Render at 640x480 (4:3 aspect ratio like real innoMaker camera)
    2. Center crop to 480x480 (1:1 square)
    3. Resize to target image_size (default 84x84)
    """

    # Proprioception dims: joint_pos(6) + joint_vel(6) + gripper_xyz(3) + gripper_euler(3)
    # Excludes cube_pos(3) which is privileged sim-only info
    PROPRIOCEPTION_DIM = 18

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] = (84, 84),
        camera: str = "wrist_cam",
    ):
        super().__init__(env)
        self.image_size = image_size
        self.camera = camera

        # Render at 640x480 to match real camera aspect ratio (4:3)
        self._render_width = 640
        self._render_height = 480
        self._crop_size = 480  # Center crop to square

        # Setup MuJoCo renderer at real camera resolution
        self._renderer = mujoco.Renderer(
            self.unwrapped.model,
            height=self._render_height,
            width=self._render_width,
        )

        # Create dict observation space with image and proprioceptive state
        # RoboBase expects (C, H, W) format, not (V, C, H, W)
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, image_size[0], image_size[1]),  # (C, H, W)
                    dtype=np.uint8,
                ),
                "low_dim_state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.PROPRIOCEPTION_DIM,),
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Convert state observation to dict with image and state.

        Preprocessing matches real camera:
        1. Render at 640x480 (4:3)
        2. Center crop to 480x480 (1:1)
        3. Resize to target size (84x84)
        """
        import cv2

        # Render from wrist camera at 640x480
        self._renderer.update_scene(self.unwrapped.data, camera=self.camera)
        img = self._renderer.render()  # (480, 640, 3)

        # Center crop to 480x480 (crop 80px from each side)
        crop_x = (self._render_width - self._crop_size) // 2  # 80
        img = img[:, crop_x:crop_x + self._crop_size, :]  # (480, 480, 3)

        # Resize to target size
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

        # Convert to channels-first (HWC -> CHW)
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)

        # Proprioception only: exclude cube_pos (last 3 dims)
        proprioception = obs[:self.PROPRIOCEPTION_DIM].astype(np.float32)

        return {"rgb": img, "low_dim_state": proprioception}

    def close(self):
        if hasattr(self, "_renderer"):
            self._renderer.close()
        super().close()


class SegmentationWrapper(gym.ObservationWrapper):
    """Outputs segmentation mask instead of RGB for sim-to-real transfer.

    5 classes:
        0: background (arm, sky, everything else)
        1: ground/table
        2: cube
        3: static finger
        4: moving finger
    """

    PROPRIOCEPTION_DIM = 18
    NUM_CLASSES = 5

    # Geom IDs from MuJoCo model
    # Note: finger pads (29, 32) are collision-only, not rendered
    # Use the visible mesh geoms instead
    FLOOR_GEOM_ID = 0
    CUBE_GEOM_ID = 33
    STATIC_FINGER_GEOM_IDS = [25, 26, 27, 28, 29]  # gripper body geoms
    MOVING_FINGER_GEOM_IDS = [30, 31, 32]  # moving_jaw body geoms

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] = (84, 84),
        camera: str = "wrist_cam",
    ):
        super().__init__(env)
        self.image_size = image_size
        self.camera = camera

        self._render_width = 640
        self._render_height = 480
        self._crop_size = 480

        self._renderer = mujoco.Renderer(
            self.unwrapped.model,
            height=self._render_height,
            width=self._render_width,
        )
        self._renderer.enable_segmentation_rendering()

        # Single-channel class IDs
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0,
                    high=self.NUM_CLASSES - 1,
                    shape=(1, image_size[0], image_size[1]),
                    dtype=np.uint8,
                ),
                "low_dim_state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.PROPRIOCEPTION_DIM,),
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        import cv2

        self._renderer.update_scene(self.unwrapped.data, camera=self.camera)
        seg = self._renderer.render()  # (H, W, 2): geom_id, geom_type

        geom_ids = seg[:, :, 0]

        # Map geom IDs to class IDs
        class_map = np.zeros_like(geom_ids, dtype=np.uint8)
        class_map[geom_ids == self.FLOOR_GEOM_ID] = 1
        class_map[geom_ids == self.CUBE_GEOM_ID] = 2
        for gid in self.STATIC_FINGER_GEOM_IDS:
            class_map[geom_ids == gid] = 3
        for gid in self.MOVING_FINGER_GEOM_IDS:
            class_map[geom_ids == gid] = 4

        # Center crop
        crop_x = (self._render_width - self._crop_size) // 2
        class_map = class_map[:, crop_x:crop_x + self._crop_size]

        # Resize
        class_map = cv2.resize(
            class_map, self.image_size, interpolation=cv2.INTER_NEAREST
        )

        # Add channel dim (H, W) -> (1, H, W)
        class_map = class_map[np.newaxis, :, :]

        proprioception = obs[:self.PROPRIOCEPTION_DIM].astype(np.float32)

        return {"rgb": class_map, "low_dim_state": proprioception}

    def close(self):
        if hasattr(self, "_renderer"):
            self._renderer.close()
        super().close()


class SegDepthWrapper(gym.ObservationWrapper):
    """Outputs 2-channel observation: segmentation mask + disparity for sim-to-real transfer.

    Channel 0: Segmentation class IDs (0-4, uint8)
        0: background (arm, sky, everything else)
        1: ground/table
        2: cube
        3: static finger
        4: moving finger

    Channel 1: Disparity (0-255, uint8)
        Matches Depth Anything V2 format:
        - 1.0 (255) = nearest pixel
        - 0.0 (0) = farthest pixel
        - Formula: disparity = 1 / depth, then normalized to [0, 1]
    """

    PROPRIOCEPTION_DIM = 18
    NUM_CLASSES = 5

    # Geom IDs from MuJoCo model
    FLOOR_GEOM_ID = 0
    CUBE_GEOM_ID = 33
    STATIC_FINGER_GEOM_IDS = [25, 26, 27, 28, 29]
    MOVING_FINGER_GEOM_IDS = [30, 31, 32]

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] = (84, 84),
        camera: str = "wrist_cam",
    ):
        super().__init__(env)
        self.image_size = image_size
        self.camera = camera

        self._render_width = 640
        self._render_height = 480
        self._crop_size = 480

        # Segmentation renderer
        self._seg_renderer = mujoco.Renderer(
            self.unwrapped.model,
            height=self._render_height,
            width=self._render_width,
        )
        self._seg_renderer.enable_segmentation_rendering()

        # Depth renderer
        self._depth_renderer = mujoco.Renderer(
            self.unwrapped.model,
            height=self._render_height,
            width=self._render_width,
        )
        self._depth_renderer.enable_depth_rendering()

        # 2-channel: seg class IDs + disparity
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(2, image_size[0], image_size[1]),
                    dtype=np.uint8,
                ),
                "low_dim_state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.PROPRIOCEPTION_DIM,),
                    dtype=np.float32,
                ),
            }
        )

    def _depth_to_disparity(self, depth: np.ndarray) -> np.ndarray:
        """Convert MuJoCo depth to disparity (inverse depth) matching Depth Anything V2 format.

        DA V2 outputs affine-invariant inverse depth normalized [0, 1] where:
        - 1 = nearest pixel
        - 0 = farthest pixel

        MuJoCo depth is [0, 1] where 0=near, 1=far.
        """
        eps = 1e-3  # Avoid division by zero
        disparity = 1.0 / (depth + eps)

        # Normalize to [0, 1] range
        d_min, d_max = disparity.min(), disparity.max()
        if d_max - d_min > 1e-6:
            disparity_norm = (disparity - d_min) / (d_max - d_min)
        else:
            disparity_norm = np.ones_like(disparity)

        return disparity_norm

    def observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        import cv2

        # --- Segmentation ---
        self._seg_renderer.update_scene(self.unwrapped.data, camera=self.camera)
        seg = self._seg_renderer.render()  # (H, W, 2): geom_id, geom_type
        geom_ids = seg[:, :, 0]

        # Map geom IDs to class IDs
        class_map = np.zeros_like(geom_ids, dtype=np.uint8)
        class_map[geom_ids == self.FLOOR_GEOM_ID] = 1
        class_map[geom_ids == self.CUBE_GEOM_ID] = 2
        for gid in self.STATIC_FINGER_GEOM_IDS:
            class_map[geom_ids == gid] = 3
        for gid in self.MOVING_FINGER_GEOM_IDS:
            class_map[geom_ids == gid] = 4

        # --- Depth -> Disparity ---
        self._depth_renderer.update_scene(self.unwrapped.data, camera=self.camera)
        depth = self._depth_renderer.render()  # (H, W), [0, 1] linear
        disparity = self._depth_to_disparity(depth)
        disparity_uint8 = (disparity * 255).astype(np.uint8)

        # --- Center crop ---
        crop_x = (self._render_width - self._crop_size) // 2
        class_map = class_map[:, crop_x:crop_x + self._crop_size]
        disparity_uint8 = disparity_uint8[:, crop_x:crop_x + self._crop_size]

        # --- Resize ---
        class_map = cv2.resize(
            class_map, self.image_size, interpolation=cv2.INTER_NEAREST
        )
        disparity_uint8 = cv2.resize(
            disparity_uint8, self.image_size, interpolation=cv2.INTER_AREA
        )

        # Stack as 2-channel (2, H, W)
        seg_depth = np.stack([class_map, disparity_uint8], axis=0)

        proprioception = obs[:self.PROPRIOCEPTION_DIM].astype(np.float32)

        return {"rgb": seg_depth, "low_dim_state": proprioception}

    def close(self):
        if hasattr(self, "_seg_renderer"):
            self._seg_renderer.close()
        if hasattr(self, "_depth_renderer"):
            self._depth_renderer.close()
        super().close()


class SO101Factory(EnvFactory):
    """Factory for SO-101 lift cube environment."""

    def _wrap_env(self, env: gym.Env, cfg: DictConfig) -> gym.Env:
        """Apply standard RoboBase wrappers."""
        # Add success info conversion for RoboBase
        env = SuccessInfoWrapper(env)

        # Rescale actions from [-1, 1] tanh output
        env = RescaleFromTanh(env)

        # Add one-hot time encoding if requested
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)

        # Add wrist camera if pixel mode
        if cfg.pixels:
            obs_type = cfg.env.get("obs_type", "rgb")
            if obs_type == "seg":
                wrapper_cls = SegmentationWrapper
            elif obs_type == "seg_depth":
                wrapper_cls = SegDepthWrapper
            else:
                wrapper_cls = WristCameraWrapper
            env = wrapper_cls(
                env,
                image_size=(cfg.env.image_size, cfg.env.image_size),
                camera="wrist_cam",
            )

        # Concatenate low-dim state along dim 1 (time axis after frame stack)
        env = ConcatDim(env, 1, 0, "low_dim_state")

        # Time limit
        env = TimeLimit(env, cfg.env.episode_length)

        # Action sequence (required by RoboBase for action chunking)
        env = ActionSequence(env, cfg.action_sequence)

        # Frame stacking
        env = FrameStack(env, cfg.frame_stack)

        # Demo info for demo-driven RL
        env = AppendDemoInfo(env)

        return env

    def _make_base_env(self, cfg: DictConfig) -> gym.Env:
        """Create base SO-101 environment."""
        return LiftCubeCartesianEnv(
            render_mode="rgb_array" if cfg.pixels else None,
            max_episode_steps=cfg.env.episode_length,
            curriculum_stage=cfg.env.get("curriculum_stage", 3),
            reward_version=cfg.env.get("reward_version", "v11"),
            lock_wrist=cfg.env.get("lock_wrist", False),
        )

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        """Create vectorized training environments."""
        return gym.vector.AsyncVectorEnv(
            [lambda: self._wrap_env(self._make_base_env(cfg), cfg) for _ in range(cfg.num_train_envs)],
            context="spawn",
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        """Create single evaluation environment."""
        return self._wrap_env(self._make_base_env(cfg), cfg)
