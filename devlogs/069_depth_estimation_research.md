# 069: Depth Estimation for Sim-to-Real

## Problem

Segmentation masks alone lack depth information. RGB has natural depth cues (shading, perspective, texture gradients) that seg masks don't have. This could make grasping harder since the agent can't judge distance to the cube.

## Options Considered

### 1. Seg + Depth (2 channels)
- Single-channel seg mask + single-channel depth
- Input: `(2, 84, 84)` per frame, `(6, 84, 84)` with frame_stack=3

### 2. RGB + Seg (4 channels)
- Keep RGB for depth cues, add seg for object identification
- Input: `(4, 84, 84)` per frame

### 3. Seg only (current)
- Rely on proprioception (gripper xyz/euler) for depth estimation
- Model knows where hand is, just needs to find cube in 2D

## Depth Estimation Models

### SOTA: Depth Anything V2

| Model | MAE | Speed | Params |
|-------|-----|-------|--------|
| Depth Anything V2 | 0.454m | 0.22s/img | 25M-1.3B |
| ZoeDepth | 3.087m | 0.17s/img | - |
| Marigold (SD-based) | worse | 10x slower | - |

**Depth Anything V2 Small (ViT-S, ~25M params)** is the best choice:
- <10ms on A100 (30+ FPS)
- Best accuracy/speed tradeoff
- Multiple sizes available (25M to 1.3B)

Source: https://arxiv.org/abs/2406.09414

### Segmentation Models

| Model | Speed (GPU) | Size | Notes |
|-------|-------------|------|-------|
| MobileSAM | ~10ms | 60x smaller than SAM | Best for edge |
| FastSAM | ~40ms | YOLO-based | No prompts needed |
| Custom trained | <5ms | ~3M params | Best for fixed classes |

For our 5-class setup, a **custom trained lightweight model** would be fastest.

Source: https://docs.ultralytics.com/models/mobile-sam/

## MuJoCo Depth Rendering

MuJoCo can render depth natively for sim training:

```python
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.enable_depth_rendering()
renderer.update_scene(data, camera="wrist_cam")
depth = renderer.render()  # float array, normalized [0, 1]

# Normalize for visualization
depth_viz = (depth - depth.min()) / (depth.max() - depth.min())
depth_viz = (depth_viz * 255).astype(np.uint8)
```

Source: https://mujoco.readthedocs.io/en/stable/python.html

## Real-Robot Pipeline

```
RGB Camera
    ├─→ Custom Seg Model (~5ms) → 5-class mask
    └─→ Depth Anything V2 Small (~10ms) → depth map
                    ↓
            Concat: (2, H, W)
                    ↓
              RL Policy
```

Total latency: ~20ms = 50 FPS

## Decision

Currently training with **seg-only** to see if proprioception compensates for lack of depth. If performance is poor, will add depth channel.

## Config

Updated configs to use explicit `obs_type` field:

```yaml
env:
  obs_type: rgb   # rgb, seg, or seg_depth (future)
```

## Commits

- `0ddb6ba` - Add obs_type config for RGB vs segmentation
- `52e84b7` - Add segmentation visualization to eval videos
