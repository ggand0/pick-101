"""Wrist camera view with RGB | Depth | Segmentation - 3-column video."""
import mujoco
import numpy as np
import imageio
from pathlib import Path
from src.envs.lift_cube import LiftCubeCartesianEnv

# 5 classes (RGB)
COLORS = {
    0: (50, 50, 50),      # background
    1: (150, 180, 200),   # ground
    2: (255, 50, 50),     # cube
    3: (50, 255, 50),     # static finger
    4: (255, 50, 255),    # moving finger
}

STATIC_FINGER_GEOM_IDS = [25, 26, 27, 28, 29]
MOVING_FINGER_GEOM_IDS = [30, 31, 32]


def geom_to_class(geom_ids):
    class_map = np.zeros_like(geom_ids, dtype=np.uint8)
    class_map[geom_ids == 0] = 1
    class_map[geom_ids == 33] = 2
    for gid in STATIC_FINGER_GEOM_IDS:
        class_map[geom_ids == gid] = 3
    for gid in MOVING_FINGER_GEOM_IDS:
        class_map[geom_ids == gid] = 4
    return class_map


def colorize(class_map):
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        rgb[class_map == class_id] = color
    return rgb


def depth_to_disparity(depth):
    """Convert MuJoCo depth to disparity (inverse depth) matching Depth Anything V2 format.

    DA V2 outputs affine-invariant inverse depth normalized [0, 1] where:
    - 1 = nearest pixel
    - 0 = farthest pixel

    MuJoCo depth is [0, 1] where 0=near, 1=far.

    Returns:
        disparity: normalized [0, 1], 1=near, 0=far (matching DA V2)
    """
    # Convert linear depth to disparity (inverse depth)
    eps = 1e-3  # Avoid division by zero
    disparity = 1.0 / (depth + eps)

    # Normalize to [0, 1] range
    d_min, d_max = disparity.min(), disparity.max()
    if d_max - d_min > 1e-6:
        disparity_norm = (disparity - d_min) / (d_max - d_min)
    else:
        disparity_norm = np.ones_like(disparity)

    return disparity_norm


def disparity_to_rgb(disparity):
    """Convert disparity to RGB visualization (matching DA V2 colormap style)."""
    # disparity is [0, 1] where 1=near (bright), 0=far (dark)
    depth_uint8 = (disparity * 255).astype(np.uint8)
    return np.stack([depth_uint8, depth_uint8, depth_uint8], axis=-1)


def main():
    env = LiftCubeCartesianEnv(render_mode="rgb_array", curriculum_stage=1)

    renderer_rgb = mujoco.Renderer(env.model, height=480, width=480)
    renderer_depth = mujoco.Renderer(env.model, height=480, width=480)
    renderer_depth.enable_depth_rendering()
    renderer_seg = mujoco.Renderer(env.model, height=480, width=480)
    renderer_seg.enable_segmentation_rendering()

    frames = []

    def capture():
        # RGB
        renderer_rgb.update_scene(env.data, camera="wrist_cam")
        rgb = renderer_rgb.render().copy()

        # Depth -> Disparity (matching Depth Anything V2 format)
        renderer_depth.update_scene(env.data, camera="wrist_cam")
        depth = renderer_depth.render()
        disparity = depth_to_disparity(depth)
        depth_colored = disparity_to_rgb(disparity)

        # Segmentation
        renderer_seg.update_scene(env.data, camera="wrist_cam")
        seg = renderer_seg.render()
        class_map = geom_to_class(seg[:, :, 0])
        seg_colored = colorize(class_map)

        return np.hstack([rgb, depth_colored, seg_colored])

    # Start with cube in gripper (stage 1)
    env.reset()
    print("Recording wrist cam segmentation...")

    # Hold lifted position
    for _ in range(50):
        env.step(np.array([0.0, 0.0, 0.0, 1.0]))
        frames.append(capture())

    # Lower
    for _ in range(80):
        env.step(np.array([0.0, 0.0, -0.5, 1.0]))
        frames.append(capture())

    # Open gripper
    for _ in range(60):
        env.step(np.array([0.0, 0.0, 0.0, -1.0]))
        frames.append(capture())

    # Rise
    for _ in range(80):
        env.step(np.array([0.0, 0.0, 0.5, -1.0]))
        frames.append(capture())

    # Move toward cube
    for _ in range(60):
        env.step(np.array([0.0, -0.5, -0.3, -1.0]))
        frames.append(capture())

    # Descend
    for _ in range(80):
        env.step(np.array([0.0, 0.0, -0.8, -1.0]))
        frames.append(capture())

    # Close
    for _ in range(60):
        env.step(np.array([0.0, 0.0, 0.0, 1.0]))
        frames.append(capture())

    # Lift
    for _ in range(100):
        env.step(np.array([0.0, 0.0, 0.8, 1.0]))
        frames.append(capture())

    # Save
    out_path = Path("outputs/wrist_cam_seg.mp4")
    out_path.parent.mkdir(exist_ok=True)
    imageio.mimsave(out_path, frames, fps=30)
    print(f"Saved {len(frames)} frames to {out_path}")

    renderer_rgb.close()
    renderer_depth.close()
    renderer_seg.close()
    env.close()


if __name__ == "__main__":
    main()
