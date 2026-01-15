"""Calculate effective FOV for camera with aspect ratio conversions.

innoMaker USB Camera specs:
- Diagonal FOV: 130°
- Horizontal FOV: 103°
- Native resolution: 1080p (1920x1080, 16:9)

Pipeline: 16:9 native → 4:3 mode (640x480) → 1:1 square crop (480x480)
"""

import numpy as np


def fov_from_focal_length(sensor_size: float, focal_length: float) -> float:
    """Calculate FOV from sensor size and focal length."""
    return 2 * np.degrees(np.arctan(sensor_size / (2 * focal_length)))


def focal_length_from_fov(sensor_size: float, fov_deg: float) -> float:
    """Calculate focal length from sensor size and FOV."""
    return sensor_size / (2 * np.tan(np.radians(fov_deg / 2)))


def crop_fov(original_fov: float, original_size: float, cropped_size: float, focal_length: float) -> float:
    """Calculate new FOV after cropping sensor."""
    return 2 * np.degrees(np.arctan(cropped_size / (2 * focal_length)))


def main():
    print("=" * 70)
    print("CAMERA FOV CALCULATION")
    print("=" * 70)
    print()

    # Camera specs
    hfov_native = 103.0  # degrees
    dfov_native = 130.0  # degrees

    # Native resolution (16:9)
    native_w, native_h = 1920, 1080

    print("INPUT: Camera specs")
    print(f"  Horizontal FOV: {hfov_native}°")
    print(f"  Diagonal FOV: {dfov_native}°")
    print(f"  Native resolution: {native_w}x{native_h} (16:9)")
    print()

    # Method 1: Calculate VFOV from HFOV using aspect ratio
    print("=" * 70)
    print("METHOD 1: VFOV from HFOV and aspect ratio")
    print("=" * 70)
    print()
    print("  tan(VFOV/2) / tan(HFOV/2) = height / width")
    print(f"  tan(VFOV/2) = tan({hfov_native/2}°) × ({native_h}/{native_w})")

    tan_hfov_half = np.tan(np.radians(hfov_native / 2))
    tan_vfov_half = tan_hfov_half * (native_h / native_w)
    vfov_from_aspect = 2 * np.degrees(np.arctan(tan_vfov_half))

    print(f"  tan(VFOV/2) = {tan_hfov_half:.4f} × {native_h/native_w:.4f} = {tan_vfov_half:.4f}")
    print(f"  VFOV = {vfov_from_aspect:.1f}°")
    print()

    # Verify with diagonal
    tan_dfov_half_calc = np.sqrt(tan_hfov_half**2 + tan_vfov_half**2)
    dfov_calc = 2 * np.degrees(np.arctan(tan_dfov_half_calc))
    print(f"  Verify diagonal: √(tan²(H/2) + tan²(V/2)) → DFOV = {dfov_calc:.1f}°")
    print(f"  Stated diagonal: {dfov_native}° (difference: {abs(dfov_calc - dfov_native):.1f}°)")
    print()

    # Method 2: Calculate VFOV from diagonal FOV
    print("=" * 70)
    print("METHOD 2: VFOV from diagonal FOV")
    print("=" * 70)
    print()
    print("  tan²(DFOV/2) = tan²(HFOV/2) + tan²(VFOV/2)")

    tan_dfov_half = np.tan(np.radians(dfov_native / 2))
    tan_vfov_half_from_diag = np.sqrt(tan_dfov_half**2 - tan_hfov_half**2)
    vfov_from_diag = 2 * np.degrees(np.arctan(tan_vfov_half_from_diag))

    print(f"  tan²({dfov_native/2}°) = tan²({hfov_native/2}°) + tan²(VFOV/2)")
    print(f"  {tan_dfov_half**2:.4f} = {tan_hfov_half**2:.4f} + tan²(VFOV/2)")
    print(f"  tan²(VFOV/2) = {tan_dfov_half**2 - tan_hfov_half**2:.4f}")
    print(f"  VFOV = {vfov_from_diag:.1f}°")
    print()

    # Check what aspect ratio this implies
    implied_aspect = tan_hfov_half / tan_vfov_half_from_diag
    print(f"  Implied aspect ratio: {implied_aspect:.3f}:1")
    print(f"  16:9 = {16/9:.3f}:1")
    print(f"  4:3 = {4/3:.3f}:1")
    print()

    print("=" * 70)
    print("CONCLUSION: Specs are inconsistent")
    print("=" * 70)
    print()
    print(f"  From aspect ratio (16:9): VFOV = {vfov_from_aspect:.1f}°")
    print(f"  From diagonal formula:   VFOV = {vfov_from_diag:.1f}°")
    print()
    print("  The diagonal FOV 130° and horizontal FOV 103° cannot both be")
    print("  correct for a 16:9 sensor. Using HFOV as ground truth.")
    print()

    # Use HFOV-derived values for pipeline calculation
    vfov_native = vfov_from_aspect

    print("=" * 70)
    print("PIPELINE: 16:9 → 4:3 → 1:1")
    print("=" * 70)
    print()

    # Calculate focal length (in pixels) from native HFOV
    f = focal_length_from_fov(native_w, hfov_native)
    print(f"Step 0: Focal length from native HFOV")
    print(f"  f = {native_w} / (2 × tan({hfov_native/2}°)) = {f:.1f} pixels")
    print()

    # Step 1: 16:9 native
    print(f"Step 1: Native 16:9 ({native_w}x{native_h})")
    print(f"  HFOV = {hfov_native:.1f}°")
    print(f"  VFOV = {vfov_native:.1f}°")
    print()

    # Step 2: Crop to 4:3 (keep full height, crop width)
    # 4:3 means width = height * 4/3
    crop_4x3_h = native_h  # 1080
    crop_4x3_w = int(crop_4x3_h * 4 / 3)  # 1440

    hfov_4x3 = fov_from_focal_length(crop_4x3_w, f)
    vfov_4x3 = vfov_native  # unchanged, no vertical crop

    print(f"Step 2: Crop to 4:3 ({crop_4x3_w}x{crop_4x3_h})")
    print(f"  Keep full height, crop width from {native_w} to {crop_4x3_w}")
    print(f"  HFOV = 2 × atan({crop_4x3_w} / (2 × {f:.1f})) = {hfov_4x3:.1f}°")
    print(f"  VFOV = {vfov_4x3:.1f}° (unchanged)")
    print()

    # Step 3: Crop to 1:1 square (crop width to match height)
    crop_1x1 = crop_4x3_h  # 1080x1080

    hfov_1x1 = fov_from_focal_length(crop_1x1, f)
    vfov_1x1 = vfov_4x3  # unchanged

    print(f"Step 3: Crop to 1:1 ({crop_1x1}x{crop_1x1})")
    print(f"  Crop width from {crop_4x3_w} to {crop_1x1}")
    print(f"  HFOV = 2 × atan({crop_1x1} / (2 × {f:.1f})) = {hfov_1x1:.1f}°")
    print(f"  VFOV = {vfov_1x1:.1f}° (unchanged)")
    print()

    print("=" * 70)
    print("RESULT: MuJoCo fovy for 1:1 square render")
    print("=" * 70)
    print()
    print(f"  fovy = {vfov_1x1:.1f}°")
    print()
    print("  Note: For a square render, HFOV = VFOV, so the horizontal")
    print(f"  content will be {hfov_1x1:.1f}° (same as VFOV).")
    print()

    # Alternative: if we want to match the HORIZONTAL content of the 1:1 crop
    print("=" * 70)
    print("ALTERNATIVE: Match horizontal FOV of cropped image")
    print("=" * 70)
    print()
    print("  If the real camera pipeline crops horizontally first,")
    print("  and we want the sim to match that horizontal extent:")
    print(f"  fovy = HFOV of 1:1 crop = {hfov_1x1:.1f}°")
    print()


if __name__ == "__main__":
    main()
