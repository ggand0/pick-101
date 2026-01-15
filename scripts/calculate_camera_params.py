"""Calculate wrist camera parameters from IRL measurements.

Coordinate system (gripper local frame):
    - X: between fingers (+ toward moving finger, - toward static finger)
    - Y: orthogonal to arm (+ toward robot body, - toward camera side)
    - Z: along arm axis (+ toward wrist, - toward fingertips)
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_look_dir(pitch_deg):
    """Get camera look direction for given pitch (with yaw=180)."""
    rot = R.from_euler("xyz", [pitch_deg, 0, 180], degrees=True)
    return rot.as_matrix() @ np.array([0, 0, -1])


def calculate_camera_params(
    B_cm: float,
    forward_cm: float,
    cam_x: float = 0.015,
    cam_z: float = -0.015,
    gripperframe_z: float = -0.0981,
):
    """Calculate camera pitch from position and target.

    Args:
        B_cm: Camera Y offset from gripper center (cm)
        forward_cm: Target forward offset from gripperframe (cm, in -Z direction toward fingertips)
        cam_x: Camera X position (m)
        cam_z: Camera Z position (m)
        gripperframe_z: Z position of gripperframe (m)
    """
    B = B_cm / 100.0
    forward = forward_cm / 100.0

    cam_y = -B
    cam_pos = np.array([cam_x, cam_y, cam_z])
    gripperframe_pos = np.array([0, 0, gripperframe_z])
    # Forward is along -Z axis (toward fingertips)
    target_pos = gripperframe_pos + np.array([0, 0, -forward])

    target_dir = target_pos - cam_pos
    target_dir = target_dir / np.linalg.norm(target_dir)

    # Find best pitch
    best_pitch = None
    best_error = float('inf')
    for pitch in np.linspace(-90, 90, 1801):
        look = get_look_dir(pitch)
        error = np.linalg.norm(look - target_dir)
        if error < best_error:
            best_error = error
            best_pitch = pitch

    print("=" * 60)
    print("INPUTS")
    print("=" * 60)
    print(f"B (camera Y offset from center): {B_cm} cm")
    print(f"Forward (target offset from gripperframe): {forward_cm} cm")
    print()

    print("=" * 60)
    print("POSITIONS (meters)")
    print("=" * 60)
    print(f"Camera:  {cam_pos}")
    print(f"Gripperframe: {gripperframe_pos}")
    print(f"Target (gripperframe + {forward_cm}cm forward): {target_pos}")
    print()

    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Target direction: {target_dir}")
    print(f"Best pitch: {best_pitch:.1f}°")
    print(f"Look direction: {get_look_dir(best_pitch)}")
    print()

    print("=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"pos = [{cam_x}, {cam_y:.4f}, {cam_z}]")
    print(f"euler = [{best_pitch:.1f}, 0, 180]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=float, required=True,
                        help="Camera Y offset from gripper center (cm)")
    parser.add_argument("--forward", type=float, required=True,
                        help="Target forward offset from gripperframe (cm)")
    parser.add_argument("--cam-x", type=float, default=0.015)
    parser.add_argument("--cam-z", type=float, default=-0.015)
    args = parser.parse_args()

    calculate_camera_params(
        B_cm=args.B,
        forward_cm=args.forward,
        cam_x=args.cam_x,
        cam_z=args.cam_z,
    )


if __name__ == "__main__":
    main()
