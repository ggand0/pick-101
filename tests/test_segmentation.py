"""Test segmentation wrapper visualization with multiple poses."""

import numpy as np
import mujoco
import cv2

from src.envs.lift_cube import LiftCubeCartesianEnv


# 5 classes (BGR for cv2)
COLORS = {
    0: (50, 50, 50),      # background - dark gray
    1: (200, 180, 150),   # ground - tan
    2: (0, 0, 255),       # cube - red
    3: (0, 255, 0),       # static finger - green
    4: (255, 0, 255),     # moving finger - magenta
}

CLASS_NAMES = ["bg", "ground", "cube", "static_finger", "moving_finger"]


def render_segmentation(env, camera="wrist_cam"):
    """Render segmentation mask from environment."""
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(env.data, camera=camera)
    seg = renderer.render()
    renderer.close()
    return seg[:, :, 0]


def render_rgb(env, camera="wrist_cam"):
    """Render RGB from environment."""
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    renderer.update_scene(env.data, camera=camera)
    rgb = renderer.render()
    renderer.close()
    return rgb


def geom_to_class(geom_ids):
    """Map geom IDs to class IDs."""
    class_map = np.zeros_like(geom_ids, dtype=np.uint8)
    class_map[geom_ids == 0] = 1   # floor
    class_map[geom_ids == 33] = 2  # cube
    # Static finger (gripper body geoms)
    for gid in [25, 26, 27, 28, 29]:
        class_map[geom_ids == gid] = 3
    # Moving finger (moving_jaw body geoms)
    for gid in [30, 31, 32]:
        class_map[geom_ids == gid] = 4
    return class_map


def colorize(class_map):
    """Convert class map to RGB image."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        rgb[class_map == class_id] = color
    return rgb


def crop_and_resize(img, crop_x=80, crop_size=480, target_size=240):
    """Center crop and resize."""
    cropped = img[:, crop_x:crop_x + crop_size]
    if len(cropped.shape) == 2:
        return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)


def render_pose(env, label):
    """Render RGB and segmentation for current pose."""
    rgb = render_rgb(env)
    geom_ids = render_segmentation(env)
    class_map = geom_to_class(geom_ids)
    seg_colored = colorize(class_map)

    rgb = crop_and_resize(rgb)
    seg_colored = crop_and_resize(seg_colored)

    # Add label
    combined = np.hstack([rgb[:, :, ::-1], seg_colored])
    cv2.putText(combined, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return combined


def main():
    frames = []

    # Pose 1: Stage 1 - cube in gripper, lifted
    env = LiftCubeCartesianEnv(render_mode="rgb_array", curriculum_stage=1)
    env.reset()
    frames.append(render_pose(env, "1. Lifted (stage 1)"))
    env.close()

    # Pose 2: Stage 2 - cube in gripper, at grasp height
    env = LiftCubeCartesianEnv(render_mode="rgb_array", curriculum_stage=2)
    env.reset()
    frames.append(render_pose(env, "2. Grasped (stage 2)"))
    env.close()

    # Pose 3: Stage 3 - gripper near cube
    env = LiftCubeCartesianEnv(render_mode="rgb_array", curriculum_stage=3)
    env.reset()
    frames.append(render_pose(env, "3. Near cube (stage 3)"))
    env.close()

    # Pose 4: Stage 0 (home), then move above cube
    env = LiftCubeCartesianEnv(render_mode="rgb_array", curriculum_stage=0)
    env.reset()
    for _ in range(40):
        env.step(np.array([0.0, -0.5, -0.3, -1.0]))  # toward + slight down
    frames.append(render_pose(env, "4. Above cube"))
    env.close()

    # Pose 5: Home position
    env = LiftCubeCartesianEnv(render_mode="rgb_array", curriculum_stage=0)
    env.reset()
    frames.append(render_pose(env, "5. Home (stage 0)"))
    env.close()

    # Stack vertically
    grid = np.vstack(frames)

    # Add legend at bottom
    legend_h = 40
    legend = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)
    x = 10
    for class_id, name in enumerate(CLASS_NAMES):
        color = COLORS[class_id]
        cv2.rectangle(legend, (x, 10), (x + 20, 30), color, -1)
        cv2.putText(legend, name, (x + 25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        x += 95

    final = np.vstack([grid, legend])

    out_path = "outputs/seg_test.png"
    cv2.imwrite(out_path, final)
    print(f"Saved to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
