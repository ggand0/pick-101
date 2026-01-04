"""Image observation wrapper for converting state-based envs to image-based."""

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


class ImageObsWrapper(gym.ObservationWrapper):
    """Wraps environment to return image observations from wrist camera.

    Converts state-based observations to camera images for visual RL.
    The wrapper renders from the wrist_cam defined in the MuJoCo XML.

    Args:
        env: Base environment (must have model and data attributes)
        image_size: Output image size (height, width). Default (84, 84)
        camera: Camera name to render from. Default "wrist_cam"
        channels_first: If True, return (C, H, W). If False, return (H, W, C).
                       Default True for PyTorch compatibility.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] = (84, 84),
        camera: str = "wrist_cam",
        channels_first: bool = True,
    ):
        super().__init__(env)
        self.image_size = image_size
        self.camera = camera
        self.channels_first = channels_first

        # Setup MuJoCo renderer
        self._renderer = mujoco.Renderer(
            self.unwrapped.model,
            height=image_size[0],
            width=image_size[1],
        )

        # Define observation space
        if channels_first:
            shape = (3, image_size[0], image_size[1])
        else:
            shape = (image_size[0], image_size[1], 3)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert state observation to camera image.

        Args:
            obs: Original state observation (ignored)

        Returns:
            Camera image as uint8 array
        """
        # Render from wrist camera
        self._renderer.update_scene(self.unwrapped.data, camera=self.camera)
        img = self._renderer.render()

        # Convert to channels-first if needed (HWC -> CHW)
        if self.channels_first:
            img = np.transpose(img, (2, 0, 1))

        return img

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and return image observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def close(self):
        """Clean up renderer."""
        if hasattr(self, "_renderer"):
            self._renderer.close()
        super().close()
