# pick-101

RL training for SO-101 robot manipulation (grasp, lift, place).

## Installation

```bash
git clone git@github.com:ggand0/pick-101.git
cd pick-101
uv sync
```

Assets (STL meshes) are stored via Git LFS and pulled automatically. Alternatively, copy from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100):
```bash
cp -r SO-ARM100/Simulation/SO101/assets models/so101/
```

## Quick Start

Run the top-down pick demo:

```bash
PYTHONPATH=. uv run python tests/test_topdown_pick.py --viewer
```

## Project Structure

```
models/so101/               # Robot models and scenes
├── so101_new_calib.xml     # Current robot with finger pads
├── lift_cube.xml           # Scene with elliptic friction
└── assets/                 # STL mesh files (Git LFS)

src/
├── controllers/
│   └── ik_controller.py    # Damped least-squares IK
├── envs/
│   └── lift_cube.py        # Gym environment with reward versions
└── training/
    ├── train_image_rl.py   # DrQ-v2 training script
    ├── eval_checkpoint.py  # Evaluation with video generation
    └── workspace.py        # SO101Workspace for RoboBase

configs/                    # Training configs
├── drqv2_lift_s3_v19.yaml  # Image-based RL (working)
└── curriculum_stage3.yaml  # State-based RL

tests/                      # Test scripts
```

## Test Scripts

| Script | Description |
|--------|-------------|
| `test_topdown_pick.py` | Current best - top-down pick with finger pads |
| `test_ik_grasp.py` | Legacy IK grasp (original model) |
| `test_horizontal_grasp.py` | Experimental horizontal approach |

### Top-Down Pick (`test_topdown_pick.py`)

The current best pick-and-place implementation. Uses a clean 4-step sequence:

1. **Move above block** - Position fingertips 30mm above cube, gripper open
2. **Descend to block** - Lower to grasp height, gripper open
3. **Close gripper** - Gradual close with contact detection, then tighten
4. **Lift** - Raise cube to target height

Key features:
- Uses `gripperframe` site at fingertips for precise IK targeting
- Finger pad collision boxes for stable multi-contact grasping
- Elliptic cone friction model to prevent slip
- Contact detection to stop closing at optimal grip force
- Locked wrist joints (`wrist_flex=90°`, `wrist_roll=90°`) for top-down orientation

```bash
# Headless
PYTHONPATH=. uv run python tests/test_topdown_pick.py

# With viewer
PYTHONPATH=. uv run python tests/test_topdown_pick.py --viewer
```

## State-Based RL (SAC)

Train an RL agent using low-dimensional state observations (joint positions, cube pose, etc.):

```bash
PYTHONPATH=. uv run python train_lift.py --config configs/curriculum_stage3.yaml
```

Uses v11 reward. Achieves 100% success rate at 1M steps.

The agent learns to:
1. Approach the cube from above
2. Close gripper to grasp
3. Lift to 8cm height
4. Hold for 3 seconds

Training outputs are saved to `runs/lift_curriculum_s3/<timestamp>/`:
- `checkpoints/` - Model checkpoints every 100k steps
- `vec_normalize.pkl` - Observation normalization stats
- `tensorboard/` - Training logs

### Evaluation

Evaluate a trained model and generate videos:

```bash
PYTHONPATH=. uv run python eval_cartesian.py \
  --run runs/lift_curriculum_s3/<timestamp> \
  --checkpoint 1000000
```

This runs 10 deterministic episodes and saves videos to the run directory.

### Resume Training

To continue training from a checkpoint:

```bash
PYTHONPATH=. uv run python train_lift.py \
  --config configs/curriculum_stage3.yaml \
  --resume runs/lift_curriculum_s3/<timestamp> \
  --timesteps 500000  # Additional steps
```

## Image-Based RL (DrQ-v2)

Train an RL agent using wrist camera observations (84x84 RGB images):

```bash
MUJOCO_GL=egl uv run python src/training/train_image_rl.py \
    --config configs/drqv2_lift_s3_v19.yaml
```

Training takes ~8 hours for 2M steps.

### Evaluate Checkpoint

```bash
MUJOCO_GL=egl uv run python src/training/eval_checkpoint.py \
    runs/image_rl/<timestamp>/snapshots/2000000_snapshot.pt \
    --num_episodes 10 \
    --reward_version v19 \
    --output_dir runs/image_rl/<timestamp>/eval
```

### X-Format Video (Side-by-Side Views)

```bash
MUJOCO_GL=egl uv run python src/training/eval_checkpoint.py \
    runs/image_rl/<timestamp>/snapshots/2000000_snapshot.pt \
    --num_episodes 5 \
    --reward_version v19 \
    --x-format \
    --output_dir runs/image_rl/<timestamp>/eval_x_post
```

Uses v19 reward with per-finger reach and hold count bonus. Achieves 100% success rate at 2M steps.

## Robot Model Variants

| Model | Features | Use Case |
|-------|----------|----------|
| `so101_new_calib.xml` | Finger pads, fingertip sites | Current development |
| `so101_ik_grasp.xml` | Original model (no pads) | Legacy compatibility |
| `so101_horizontal_grasp.xml` | Horizontal approach config | Experimental |
