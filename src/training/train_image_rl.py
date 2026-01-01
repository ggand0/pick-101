"""Train image-based RL agent for SO-101 lift task using RoboBase.

Resume training:
    MUJOCO_GL=egl uv run python src/training/train_image_rl.py \
        resume_from=runs/image_rl/20231231_120000/snapshots/latest_snapshot.pt
"""

# Use spawn for multiprocessing (required for EGL/GPU rendering on AMD)
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig

from src.training.workspace import SO101Workspace
from src.training.so101_factory import SO101Factory


# Find RoboBase config path
import robobase
ROBOBASE_CFG_PATH = str(Path(robobase.__file__).parent / "cfgs")


@hydra.main(
    config_path="cfgs",
    config_name="so101_lift",
    version_base=None,
)
def main(cfg: DictConfig):
    """Train image-based RL agent."""
    # Create workspace with SO-101 factory
    workspace = SO101Workspace(cfg, env_factory=SO101Factory())

    # Check for resume
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        snapshot_path = Path(resume_from)
        if snapshot_path.exists():
            print(f"Resuming from snapshot: {snapshot_path}")
            workspace.load_snapshot(snapshot_path)
        else:
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    workspace.train()


if __name__ == "__main__":
    # Register RoboBase config path with hydra
    from hydra import initialize_config_dir
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    class RoboBaseSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(self, search_path):
            search_path.prepend(
                provider="robobase",
                path=f"file://{ROBOBASE_CFG_PATH}",
            )

    Plugins.instance().register(RoboBaseSearchPathPlugin)
    main()
