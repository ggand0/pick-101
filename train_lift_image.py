"""Train image-based lift policy using SAC with CNN.

Uses wrist camera observations instead of state for sim-to-real transfer.
Provides familiar SB3-style logging with tqdm progress bar.

Resume training:
    python train_lift_image.py --resume runs/lift_image/20231231_120000
"""
import argparse
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from src.callbacks.plot_callback import PlotLearningCurveCallback
from src.envs.lift_cube import LiftCubeCartesianEnv
from src.envs.wrappers.image_obs import ImageObsWrapper


class ReplayBufferCheckpointCallback(BaseCallback):
    """Save replay buffer alongside model checkpoints."""

    def __init__(self, save_path: str, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            replay_path = self.save_path / f"replay_buffer_{self.num_timesteps}.pkl"
            with open(replay_path, "wb") as f:
                pickle.dump(self.model.replay_buffer, f)
            if self.verbose > 0:
                print(f"Saved replay buffer to {replay_path}")
            # Keep only latest replay buffer to save disk space
            for old_file in self.save_path.glob("replay_buffer_*.pkl"):
                if old_file != replay_path:
                    old_file.unlink()
        return True


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict, image_cfg: dict):
    """Create environment with image observations."""
    env = LiftCubeCartesianEnv(
        render_mode=None,
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        hold_steps=env_cfg.get("hold_steps", 10),
        reward_type=env_cfg.get("reward_type", "dense"),
        reward_version=env_cfg.get("reward_version", "v11"),
        curriculum_stage=env_cfg.get("curriculum_stage", 3),
        lock_wrist=env_cfg.get("lock_wrist", True),
    )

    # Wrap with image observations
    env = ImageObsWrapper(
        env,
        image_size=(image_cfg.get("size", 84), image_cfg.get("size", 84)),
        camera=image_cfg.get("camera", "wrist_cam"),
        channels_first=False,  # SB3 expects HWC, VecTransposeImage converts to CHW
    )

    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lift_image.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    env_cfg = config["env"]
    image_cfg = config.get("image", {"size": 84, "camera": "wrist_cam", "frame_stack": 3})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise ValueError(f"Resume directory not found: {resume_dir}")
        # Continue in the SAME directory for seamless resume
        output_dir = resume_dir
        print(f"Resuming in existing directory: {output_dir}")
    else:
        resume_dir = None
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        # Copy config only for new runs
        shutil.copy(args.config, output_dir / "config.yaml")

    # Create vectorized environments with image observations
    env = DummyVecEnv([lambda: make_env(env_cfg, image_cfg)])
    # Frame stacking for temporal information
    env = VecFrameStack(env, n_stack=image_cfg.get("frame_stack", 3))
    # Convert HWC to CHW for PyTorch CNN
    env = VecTransposeImage(env)

    eval_env = DummyVecEnv([lambda: make_env(env_cfg, image_cfg)])
    eval_env = VecFrameStack(eval_env, n_stack=image_cfg.get("frame_stack", 3))
    eval_env = VecTransposeImage(eval_env)

    # Create or load model
    resume_step = 0
    if args.resume:
        checkpoints = list((output_dir / "checkpoints").glob("*.zip"))
        def get_step_number(path):
            name = path.stem
            parts = name.split("_")
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 0
        checkpoints = sorted(checkpoints, key=get_step_number)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            resume_step = get_step_number(latest_checkpoint)
            model = SAC.load(latest_checkpoint, env=env, device=device)
            model.tensorboard_log = str(output_dir / "tensorboard")
            print(f"Resumed from {latest_checkpoint} (step {resume_step})")

            # Try to load replay buffer if available
            replay_files = list((output_dir / "checkpoints").glob("replay_buffer_*.pkl"))
            if replay_files:
                latest_replay = max(replay_files, key=lambda p: get_step_number(p))
                print(f"Loading replay buffer from {latest_replay}...")
                with open(latest_replay, "rb") as f:
                    model.replay_buffer = pickle.load(f)
                print(f"Loaded {model.replay_buffer.size()} transitions")
            else:
                print("No replay buffer found, starting fresh (will refill during training)")
        else:
            raise ValueError(f"No checkpoints found in {output_dir / 'checkpoints'}")
    else:
        # CNN policy for image observations
        policy_kwargs = dict(
            net_arch=dict(
                pi=sac_cfg.get("policy_layers", [256, 256]),
                qf=sac_cfg.get("qf_layers", [256, 256]),
            ),
        )

        model = SAC(
            "CnnPolicy",
            env,
            learning_rate=sac_cfg.get("learning_rate", 3e-4),
            buffer_size=sac_cfg.get("buffer_size", 100000),
            learning_starts=sac_cfg.get("learning_starts", 1000),
            batch_size=sac_cfg.get("batch_size", 256),
            tau=sac_cfg.get("tau", 0.005),
            gamma=sac_cfg.get("gamma", 0.99),
            train_freq=sac_cfg.get("train_freq", 1),
            gradient_steps=sac_cfg.get("gradient_steps", 1),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=train_cfg.get("seed", 1),
            device=device,
            tensorboard_log=str(output_dir / "tensorboard"),
        )

    # Callbacks
    save_freq = train_cfg.get("save_freq", 50000)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_image",
    )

    replay_buffer_callback = ReplayBufferCheckpointCallback(
        save_path=str(output_dir / "checkpoints"),
        save_freq=save_freq,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_cfg.get("eval_freq", 10000),
        deterministic=True,
        render=False,
    )

    plot_callback = PlotLearningCurveCallback(
        run_dir=output_dir,
        save_freq=save_freq,
        verbose=1,
        resume_step=resume_step,
    )

    timesteps = args.timesteps if args.timesteps is not None else train_cfg.get("timesteps", 1000000)

    if args.resume:
        print(f"Loaded model num_timesteps: {model.num_timesteps}")
        reset_num_timesteps = False
        learn_timesteps = timesteps
        target_total = model.num_timesteps + timesteps
        print(f"\nResuming Image Lift training from step {model.num_timesteps}...")
        print(f"Training for {timesteps} additional timesteps (target: {target_total} total)")
    else:
        reset_num_timesteps = True
        learn_timesteps = timesteps
        target_total = timesteps
        print(f"\nStarting Image Lift training for {timesteps} timesteps...")

    print(f"Image size: {image_cfg.get('size', 84)}x{image_cfg.get('size', 84)}")
    print(f"Frame stack: {image_cfg.get('frame_stack', 3)}")
    print(f"Camera: {image_cfg.get('camera', 'wrist_cam')}")
    print(f"Output directory: {output_dir}")

    model.learn(
        total_timesteps=learn_timesteps,
        callback=[checkpoint_callback, replay_buffer_callback, eval_callback, plot_callback],
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(output_dir / "final_model")

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
