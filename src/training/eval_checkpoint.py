"""Evaluate a trained checkpoint and generate multi-camera videos.

Usage:
    MUJOCO_GL=egl uv run python src/training/eval_checkpoint.py \
        runs/image_rl/20251231_145806/snapshots/400000_snapshot.pt \
        --num_episodes 5 \
        --output_dir runs/image_rl/20251231_145806/eval_videos
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from src.envs.lift_cube import LiftCubeCartesianEnv


def create_video_writer(path: Path, fps: int = 30):
    """Create video writer using imageio."""
    import imageio
    return imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8)


def render_multi_camera(env: LiftCubeCartesianEnv, cameras: list[str], size: int = 256) -> dict[str, np.ndarray]:
    """Render from multiple camera views.

    Supports named cameras (e.g., 'wrist_cam') and virtual views:
    - 'topdown': Bird's eye view
    - 'side': Side view
    - 'front': Front view
    - 'iso': Isometric view
    """
    frames = {}
    renderer = mujoco.Renderer(env.model, height=size, width=size)

    # Virtual camera configurations: (lookat, distance, azimuth, elevation)
    virtual_cameras = {
        "topdown": ([0.35, 0.0, 0.0], 0.8, 90, -90),  # Bird's eye
        "side": ([0.35, 0.0, 0.1], 0.7, 0, -15),      # Side view
        "front": ([0.35, 0.0, 0.1], 0.7, 90, -15),    # Front view
        "iso": ([0.35, 0.0, 0.1], 0.8, 135, -30),     # Isometric view
    }

    for cam in cameras:
        if cam in virtual_cameras:
            # Virtual camera - create MjvCamera struct
            lookat, dist, azim, elev = virtual_cameras[cam]
            cam_obj = mujoco.MjvCamera()
            cam_obj.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam_obj.lookat[:] = lookat
            cam_obj.distance = dist
            cam_obj.azimuth = azim
            cam_obj.elevation = elev
            renderer.update_scene(env.data, camera=cam_obj)
        else:
            # Named camera from model
            try:
                renderer.update_scene(env.data, camera=cam)
            except Exception as e:
                print(f"Warning: Camera '{cam}' not found, using default view")
                renderer.update_scene(env.data)

        frames[cam] = renderer.render().copy()

    renderer.close()
    return frames


def combine_frames(frames: dict[str, np.ndarray], layout: str = "horizontal") -> np.ndarray:
    """Combine multiple camera frames into a single image."""
    frame_list = list(frames.values())

    if layout == "horizontal":
        return np.concatenate(frame_list, axis=1)
    elif layout == "vertical":
        return np.concatenate(frame_list, axis=0)
    elif layout == "grid":
        # 2x2 grid for 4 cameras
        n = len(frame_list)
        if n == 4:
            top = np.concatenate(frame_list[:2], axis=1)
            bottom = np.concatenate(frame_list[2:], axis=1)
            return np.concatenate([top, bottom], axis=0)
        else:
            return np.concatenate(frame_list, axis=1)
    else:
        return np.concatenate(frame_list, axis=1)


def evaluate_with_video(
    snapshot_path: Path,
    num_episodes: int = 5,
    output_dir: Path = None,
    cameras: list[str] = None,
    frame_size: int = 256,
    curriculum_stage: int = 3,
    reward_version: str = "v11",
    save_video: bool = True,
    debug: bool = False,
):
    """Evaluate checkpoint and save multi-camera videos."""

    if cameras is None:
        cameras = ["topdown", "wrist_cam", "side", "front"]

    if output_dir is None:
        output_dir = snapshot_path.parent.parent / "eval_videos"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load snapshot to get config
    with open(snapshot_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", weights_only=False)

    cfg = payload["cfg"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import required modules
    import hydra
    from gymnasium.wrappers import TimeLimit
    from src.training.so101_factory import WristCameraWrapper, SuccessInfoWrapper
    from robobase.envs.wrappers import RescaleFromTanh, FrameStack, ActionSequence, ConcatDim

    # Create base environment
    env = LiftCubeCartesianEnv(
        render_mode="rgb_array",
        max_episode_steps=cfg.env.episode_length,
        curriculum_stage=curriculum_stage,
        reward_version=reward_version,
    )

    # Wrap for agent compatibility
    env = SuccessInfoWrapper(env)
    env = RescaleFromTanh(env)
    env = WristCameraWrapper(env, image_size=(84, 84))
    env = ConcatDim(env, 1, 0, "low_dim_state")
    env = TimeLimit(env, cfg.env.episode_length)
    env = ActionSequence(env, cfg.action_sequence)
    env = FrameStack(env, cfg.frame_stack)

    # Get observation and action space info (format expected by robobase)
    obs_space = env.observation_space
    act_space = env.action_space

    observation_space = {
        "rgb": {"shape": obs_space["rgb"].shape},
        "low_dim_state": {"shape": obs_space["low_dim_state"].shape},
    }
    action_space = {
        "shape": act_space.shape,
        "minimum": float(act_space.low.flatten()[0]),
        "maximum": float(act_space.high.flatten()[0]),
    }

    # Create agent using hydra (same as workspace)
    agent = hydra.utils.instantiate(
        cfg.method,
        device=device,
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=cfg.num_train_envs,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        intrinsic_reward_module=None,
    )
    agent.load_state_dict(payload["agent"])
    agent.train(False)

    # Run evaluation episodes
    results = []

    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")

        # Video writers for this episode
        writer = None
        if save_video:
            video_path = output_dir / f"episode_{ep:02d}.mp4"
            writer = create_video_writer(video_path)

        obs, info = env.reset()
        done = False
        total_reward = 0
        ep_step = 0
        success = False

        frames_collected = []

        # Debug logging data
        debug_data = {
            "cube_z": [],
            "step_reward": [],
            "is_grasping": [],
            "gripper_pos": [],
        }

        pbar = tqdm(total=cfg.env.episode_length, desc=f"Ep {ep+1}")

        while not done:
            # Get action from agent (step=1e6 for eval since exploration schedule decays)
            with torch.no_grad():
                obs_tensor = {
                    "rgb": torch.from_numpy(obs["rgb"]).unsqueeze(0).float().to(device),
                    "low_dim_state": torch.from_numpy(obs["low_dim_state"]).unsqueeze(0).float().to(device),
                }
                action = agent.act(obs_tensor, step=1_000_000, eval_mode=True)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                # Remove batch dim, keep action_sequence dim: (1, seq, act_dim) -> (seq, act_dim)
                action = action.squeeze(0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            ep_step += 1

            # Debug logging
            if debug:
                base_env = env.unwrapped
                cube_pos = base_env.data.sensor("cube_pos").data.copy()
                cube_z = cube_pos[2]
                is_grasping = base_env._is_grasping()
                gripper_pos = base_env.ik.get_ee_position()
                debug_data["cube_z"].append(cube_z)
                debug_data["step_reward"].append(reward)
                debug_data["is_grasping"].append(is_grasping)
                debug_data["gripper_pos"].append(gripper_pos)

            # Render multi-camera view
            if save_video:
                base_env = env.unwrapped
                cam_frames = render_multi_camera(base_env, cameras, frame_size)
                combined = combine_frames(cam_frames, layout="grid" if len(cameras) == 4 else "horizontal")
                frames_collected.append(combined)

            if info.get("is_success", False):
                success = True

            pbar.update(1)

        pbar.close()

        # Write video
        if save_video and writer is not None:
            for frame in frames_collected:
                writer.append_data(frame)
            writer.close()

        # Print debug summary for this episode
        if debug:
            cube_z_arr = np.array(debug_data["cube_z"])
            rewards_arr = np.array(debug_data["step_reward"])
            grasping_arr = np.array(debug_data["is_grasping"])

            print(f"\n  --- Debug Summary ---")
            print(f"  Cube Z: min={cube_z_arr.min():.4f}, max={cube_z_arr.max():.4f}, final={cube_z_arr[-1]:.4f}")
            print(f"  Rewards: min={rewards_arr.min():.2f}, max={rewards_arr.max():.2f}, mean={rewards_arr.mean():.2f}")
            print(f"  Grasping: {grasping_arr.sum()}/{len(grasping_arr)} steps ({100*grasping_arr.mean():.1f}%)")

            # Find when cube first reaches threshold heights
            for threshold in [0.02, 0.04, 0.06, 0.08]:
                above = np.where(cube_z_arr > threshold)[0]
                if len(above) > 0:
                    print(f"  Cube > {threshold:.2f}m: first at step {above[0]}, {len(above)} total steps")

            # Save debug log to file
            debug_log_path = output_dir / f"episode_{ep:02d}_debug.txt"
            with open(debug_log_path, "w") as f:
                f.write("step,cube_z,reward,is_grasping,gripper_x,gripper_y,gripper_z\n")
                for i in range(len(debug_data["cube_z"])):
                    gp = debug_data["gripper_pos"][i]
                    f.write(f"{i},{debug_data['cube_z'][i]:.6f},{debug_data['step_reward'][i]:.4f},"
                           f"{int(debug_data['is_grasping'][i])},{gp[0]:.6f},{gp[1]:.6f},{gp[2]:.6f}\n")
            print(f"  Debug log saved: {debug_log_path}")

        results.append({
            "episode": ep,
            "reward": total_reward,
            "steps": ep_step,
            "success": success,
        })

        print(f"  Reward: {total_reward:.2f}, Steps: {ep_step}, Success: {success}")

    env.close()

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    rewards = [r["reward"] for r in results]
    successes = [r["success"] for r in results]
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Success Rate: {np.mean(successes)*100:.1f}%")
    print(f"Videos saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with multi-camera video")
    parser.add_argument("snapshot", type=str, help="Path to snapshot file")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for videos")
    parser.add_argument("--cameras", nargs="+", default=["topdown", "wrist_cam", "side", "front"],
                       help="Camera names to render")
    parser.add_argument("--frame_size", type=int, default=256, help="Frame size for rendering")
    parser.add_argument("--curriculum_stage", type=int, default=3, help="Curriculum stage")
    parser.add_argument("--reward_version", type=str, default="v11", help="Reward version")
    parser.add_argument("--no_video", action="store_true", help="Skip video generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (cube_z, rewards, grasping)")

    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(f"Error: Snapshot not found: {snapshot_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    evaluate_with_video(
        snapshot_path=snapshot_path,
        num_episodes=args.num_episodes,
        output_dir=output_dir,
        cameras=args.cameras,
        frame_size=args.frame_size,
        curriculum_stage=args.curriculum_stage,
        reward_version=args.reward_version,
        save_video=not args.no_video,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
