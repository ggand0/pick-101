# 070: Disparity Format for Sim-to-Real Depth

## Problem

MuJoCo depth and Depth Anything V2 use different representations. For sim-to-real transfer, they must match exactly.

## MuJoCo Depth Buffer

```python
renderer.enable_depth_rendering()
depth = renderer.render()  # [0, 1], 0=near, 1=far (linear depth)
```

## Depth Anything V2 Output

DA V2 outputs **affine-invariant inverse depth (disparity)**:
- Normalized to [0, 1]
- **1 = nearest pixel**
- **0 = farthest pixel**
- Formula: `disparity = 1 / depth`

Source: https://github.com/DepthAnything/Depth-Anything-V2

## Conversion

To match DA V2 format in sim:

```python
def depth_to_disparity(depth):
    """Convert MuJoCo linear depth to disparity (DA V2 format).

    Args:
        depth: MuJoCo depth [0, 1] where 0=near, 1=far

    Returns:
        disparity: [0, 1] where 1=near, 0=far (matching DA V2)
    """
    eps = 1e-3  # Avoid division by zero
    disparity = 1.0 / (depth + eps)

    # Normalize to [0, 1]
    d_min, d_max = disparity.min(), disparity.max()
    disparity_norm = (disparity - d_min) / (d_max - d_min)

    return disparity_norm
```

## Why This Matters

| Property | MuJoCo Linear | DA V2 Disparity |
|----------|---------------|-----------------|
| Near objects | Low values (0) | High values (1) |
| Far objects | High values (1) | Low values (0) |
| Scale | Linear | Inverse (1/d) |

If sim outputs linear depth but real uses disparity, the policy won't transfer because:
- Relative depth relationships are inverted
- Non-linear vs linear scaling affects gradients differently

## Test Video

Updated `tests/test_wrist_cam_seg_video.py` to output 3-column video:
- RGB | Disparity | Segmentation
- Disparity now uses proper `1/depth` conversion matching DA V2

Output: `outputs/wrist_cam_seg.mp4`

## Future Work

When adding depth channel to training:
1. Use `obs_type: seg_depth`
2. Output 2 channels: seg class IDs + disparity
3. Disparity must use this conversion for sim-to-real compatibility
