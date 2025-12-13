"""Train a pick-and-place policy using SAC + HER."""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.pick_cube_goal import PickCubeGoalEnv


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict):
    """Create the GoalEnv environment."""
    return PickCubeGoalEnv(
        render_mode=None,
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.1),
        reward_type=env_cfg.get("reward_type", "sparse"),
        distance_threshold=env_cfg.get("distance_threshold", 0.03),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/her_1m.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment dir to resume")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    her_cfg = config["her"]
    env_cfg = config["env"]

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            raise ValueError(f"Resume directory not found: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output dir
    if not args.resume:
        shutil.copy(args.config, output_dir / "config.yaml")

    # Create environments
    env = DummyVecEnv([lambda: make_env(env_cfg)])
    vec_normalize_path = output_dir / "vec_normalize.pkl"

    if args.resume and vec_normalize_path.exists():
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = True
        print(f"Loaded normalization stats from {vec_normalize_path}")
    else:
        env = VecNormalize(
            env,
            norm_obs=env_cfg["normalize_obs"],
            norm_reward=env_cfg["normalize_reward"],
        )

    eval_env = DummyVecEnv([lambda: make_env(env_cfg)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=env_cfg["normalize_obs"],
        norm_reward=False,
        training=False,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=str(output_dir / "checkpoints"),
        name_prefix="her_sac_pick_cube",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_cfg["eval_freq"],
        deterministic=True,
        render=False,
    )

    # Create or load SAC + HER agent
    if args.resume:
        checkpoints = sorted((output_dir / "checkpoints").glob("*.zip"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            # HER models need env for compute_reward
            model = SAC.load(latest_checkpoint, env=env, device=device)
            model.tensorboard_log = str(output_dir / "tensorboard")
            print(f"Resumed from {latest_checkpoint}")
        else:
            raise ValueError(f"No checkpoints found in {output_dir / 'checkpoints'}")
    else:
        model = SAC(
            "MultiInputPolicy",  # Required for Dict observation space
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=her_cfg["n_sampled_goal"],
                goal_selection_strategy=her_cfg["goal_selection_strategy"],
            ),
            learning_rate=sac_cfg["learning_rate"],
            buffer_size=sac_cfg["buffer_size"],
            learning_starts=sac_cfg["learning_starts"],
            batch_size=sac_cfg["batch_size"],
            tau=sac_cfg["tau"],
            gamma=sac_cfg["gamma"],
            train_freq=sac_cfg["train_freq"],
            gradient_steps=sac_cfg["gradient_steps"],
            verbose=1,
            seed=train_cfg["seed"],
            device=device,
            tensorboard_log=str(output_dir / "tensorboard"),
        )

    print(f"\nStarting HER+SAC training for {train_cfg['timesteps']} timesteps...")
    print(f"HER config: n_sampled_goal={her_cfg['n_sampled_goal']}, strategy={her_cfg['goal_selection_strategy']}")
    print(f"Output directory: {output_dir}")

    # Train
    model.learn(
        total_timesteps=train_cfg["timesteps"],
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model and normalization stats
    model.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
