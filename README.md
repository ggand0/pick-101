# explore-mujoco

MuJoCo simulation environment for SO-101 robot arm manipulation tasks.

## Installation

```bash
git clone <repo-url>
cd explore-mujoco
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
models/so101/           # Robot models and scenes
├── so101_new_calib.xml # Current robot with finger pads
├── lift_cube.xml       # Scene with elliptic friction
└── assets/             # STL mesh files (Git LFS)

src/controllers/
└── ik_controller.py    # Damped least-squares IK

tests/                  # Test scripts
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

## Robot Model Variants

| Model | Features | Use Case |
|-------|----------|----------|
| `so101_new_calib.xml` | Finger pads, fingertip sites | Current development |
| `so101_ik_grasp.xml` | Original model (no pads) | Legacy compatibility |
| `so101_horizontal_grasp.xml` | Horizontal approach config | Experimental |
