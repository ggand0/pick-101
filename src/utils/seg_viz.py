"""Segmentation visualization utilities."""
import numpy as np

# 5 classes (RGB)
SEG_COLORS = {
    0: (50, 50, 50),      # background - dark gray
    1: (150, 180, 200),   # ground - tan/blue
    2: (255, 50, 50),     # cube - red
    3: (50, 255, 50),     # static finger - green
    4: (255, 50, 255),    # moving finger - magenta
}

SEG_CLASS_NAMES = ["background", "ground", "cube", "static_finger", "moving_finger"]


def colorize_segmentation(class_map: np.ndarray) -> np.ndarray:
    """Convert single-channel class map to RGB image.

    Args:
        class_map: (H, W) or (1, H, W) array of class IDs (0-4)

    Returns:
        (H, W, 3) RGB image
    """
    if class_map.ndim == 3:
        class_map = class_map[0]  # Remove channel dim

    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in SEG_COLORS.items():
        rgb[class_map == class_id] = color

    return rgb


def colorize_stacked_frames(stacked: np.ndarray) -> np.ndarray:
    """Colorize stacked segmentation frames.

    Args:
        stacked: (N, H, W) stacked frames where N is frame_stack count

    Returns:
        (N, H, W, 3) RGB images
    """
    n_frames = stacked.shape[0]
    h, w = stacked.shape[1], stacked.shape[2]
    rgb_frames = np.zeros((n_frames, h, w, 3), dtype=np.uint8)

    for i in range(n_frames):
        rgb_frames[i] = colorize_segmentation(stacked[i])

    return rgb_frames
