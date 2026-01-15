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
    A_cm: float,
    B_cm: float,
    forward_cm: float = 0.0,
    cam_x: float = 0.01,
    gripperframe_z: float = -0.0981,
):
    """Calculate camera pitch from position and target.

    Args:
        A_cm: Vertical distance from gripperframe to camera (cm, along +Z toward wrist)
        B_cm: Camera Y offset from gripper center (cm)
        forward_cm: Optional target forward offset from gripperframe (cm, in -Z direction)
        cam_x: Camera X position (m)
        gripperframe_z: Z position of gripperframe (m)
    """
    A = A_cm / 100.0
    B = B_cm / 100.0
    forward = forward_cm / 100.0

    cam_y = -B
    cam_z = gripperframe_z + A  # Camera is A cm above gripperframe
    cam_pos = np.array([cam_x, cam_y, cam_z])
    gripperframe_pos = np.array([0, 0, gripperframe_z])
    target_pos = gripperframe_pos + np.array([0, 0, -forward])

    target_dir = target_pos - cam_pos
    target_dir = target_dir / np.linalg.norm(target_dir)

    # Calculate pitch using arctan
    # With yaw=180°, camera looks along +Y in world frame when pitch=0
    # Pitch rotates in the Y-Z plane
    best_pitch = -np.degrees(np.arctan2(target_dir[1], -target_dir[2]))

    print("=" * 60)
    print("INPUTS")
    print("=" * 60)
    print(f"A (vertical from gripperframe to camera): {A_cm} cm")
    print(f"B (camera Y offset from center): {B_cm} cm")
    if forward_cm != 0:
        print(f"Forward (target offset from gripperframe): {forward_cm} cm")
    print()

    print("=" * 60)
    print("POSITIONS (meters)")
    print("=" * 60)
    print(f"Camera:  {cam_pos}")
    print(f"Gripperframe: {gripperframe_pos}")
    print(f"Target: {target_pos}")
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
    parser.add_argument("--A", type=float, required=True,
                        help="Vertical distance from gripperframe to camera (cm)")
    parser.add_argument("--B", type=float, required=True,
                        help="Camera Y offset from gripper center (cm)")
    parser.add_argument("--forward", type=float, default=0.0,
                        help="Target forward offset from gripperframe (cm)")
    parser.add_argument("--cam-x", type=float, default=0.01)
    args = parser.parse_args()

    calculate_camera_params(
        A_cm=args.A,
        B_cm=args.B,
        forward_cm=args.forward,
        cam_x=args.cam_x,
    )


if __name__ == "__main__":
    main()
