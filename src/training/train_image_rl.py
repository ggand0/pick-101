"""Train image-based RL agent for SO-101 lift task using RoboBase.

Usage:
    MUJOCO_GL=egl uv run python src/training/train_image_rl.py --config configs/drqv2_lift_s3.yaml

Resume training:
    MUJOCO_GL=egl uv run python src/training/train_image_rl.py \
        --config configs/drqv2_lift_s3.yaml \
        --resume runs/image_rl/20231231_120000/snapshots/latest_snapshot.pt
"""

# Use spawn for multiprocessing (required for EGL/GPU rendering on AMD)
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.config_loader import load_config, instantiate
from src.training.so101_factory import SO101Factory

# Monkey-patch hydra.utils.instantiate to use our version
import hydra.utils
hydra.utils.instantiate = instantiate


def main():
    parser = argparse.ArgumentParser(description="Train image-based RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/drqv2_lift_s3.yaml",
        help="Path to config yaml file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to snapshot file to resume from",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("experiment_name", "image_rl")
    work_dir = Path("runs") / exp_name / timestamp
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output directory
    shutil.copy(args.config, work_dir / "config.yaml")

    # Import custom workspace with MultiCameraVideoRecorder (wrist_cam | closeup | wide)
    from src.training.workspace import SO101Workspace

    # Create workspace with SO-101 factory
    workspace = SO101Workspace(
        cfg=cfg,
        env_factory=SO101Factory(),
        work_dir=str(work_dir),
    )

    # Check for resume
    if args.resume:
        snapshot_path = Path(args.resume)
        if snapshot_path.exists():
            print(f"Resuming from snapshot: {snapshot_path}")
            workspace.load_snapshot(snapshot_path)
        else:
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    workspace.train()


if __name__ == "__main__":
    main()
