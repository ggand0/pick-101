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


def add_camera_markers(scene, cam_world_pos: np.ndarray, cam_dir: np.ndarray):
    """Add 3D markers to scene showing camera position and view direction.

    Adds:
    - Red sphere at camera position
    - Green sphere at view target (5cm along view direction)
    - Green cylinder connecting them
    """
    # Target point 5cm along view direction
    view_target = cam_world_pos + cam_dir * 0.05
    line_center = (cam_world_pos + view_target) / 2
    line_length = np.linalg.norm(view_target - cam_world_pos)

    # Rotation matrix to align cylinder with direction
    z_axis = cam_dir / np.linalg.norm(cam_dir)
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
    else:
        x_axis = np.cross(np.array([1, 0, 0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])

    # Red sphere at camera position
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.008, 0, 0]),
        cam_world_pos,
        np.eye(3).flatten(),
        np.array([1, 0, 0, 1]),
    )
    scene.ngeom += 1

    # Green sphere at view target
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.006, 0, 0]),
        view_target,
        np.eye(3).flatten(),
        np.array([0, 1, 0, 1]),
    )
    scene.ngeom += 1

    # Green cylinder connecting them
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.003, line_length / 2, 0]),
        line_center,
        rot_mat.flatten(),
        np.array([0, 1, 0, 0.8]),
    )
    scene.ngeom += 1


def render_multi_camera(env: LiftCubeCartesianEnv, cameras: list[str], size: int = 256, show_cam_markers: bool = False) -> dict[str, np.ndarray]:
    """Render from multiple camera views.

    Supports named cameras (e.g., 'wrist_cam') and virtual views:
    - 'topdown': Bird's eye view
    - 'side': Side view
    - 'front': Front view
    - 'iso': Isometric view

    Args:
        env: The environment to render from
        cameras: List of camera names to render
        size: Output image size (square)
        show_cam_markers: If True, show wrist camera position/direction markers

    Note: wrist_cam uses real camera preprocessing (640x480 -> 480x480 center crop -> size)
    """
    import cv2

    frames = {}
    renderer = mujoco.Renderer(env.model, height=size, width=size)
    wrist_renderer = None  # Lazy init for wrist cam

    # Get wrist camera world position and direction for markers
    cam_world_pos = None
    cam_dir = None
    if show_cam_markers:
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        if cam_id >= 0:
            cam_world_pos = env.data.cam_xpos[cam_id].copy()
            cam_world_mat = env.data.cam_xmat[cam_id].reshape(3, 3)
            cam_dir = -cam_world_mat[:, 2]  # Camera looks along -Z

    # Virtual camera configurations: (lookat, distance, azimuth, elevation)
    # Matches state-based eval videos from lift_cube.py
    virtual_cameras = {
        "closeup": ([0.40, -0.10, 0.03], 0.35, 90, -15),  # Side view close to cube
        "wide": ([0.25, -0.05, 0.05], 0.8, 135, -25),     # Diagonal view of arm and cube
        "wide2": ([0.25, -0.05, 0.05], 0.8, 45, -25),     # Diagonal from other side
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
            # Add camera markers to external views
            if cam_world_pos is not None and cam_dir is not None:
                add_camera_markers(renderer._scene, cam_world_pos, cam_dir)
            frames[cam] = renderer.render().copy()
        elif cam == "wrist_cam":
            # Wrist cam with real camera preprocessing (640x480 -> 480x480 center crop)
            if wrist_renderer is None:
                wrist_renderer = mujoco.Renderer(env.model, height=480, width=640)
            wrist_renderer.update_scene(env.data, camera="wrist_cam")
            img = wrist_renderer.render()  # (480, 640, 3)
            # Center crop to 480x480
            crop_x = (640 - 480) // 2  # 80
            img = img[:, crop_x:crop_x + 480, :]  # (480, 480, 3)
            # Resize to target size
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            frames[cam] = img
        else:
            # Named camera from model
            try:
                renderer.update_scene(env.data, camera=cam)
            except Exception as e:
                print(f"Warning: Camera '{cam}' not found, using default view")
                renderer.update_scene(env.data)
            frames[cam] = renderer.render().copy()

    renderer.close()
    if wrist_renderer is not None:
        wrist_renderer.close()
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


def render_x_format(env: LiftCubeCartesianEnv, show_cam_markers: bool = False) -> np.ndarray:
    """Render 16:9 format for X/Twitter: wrist_cam (720x720) + wide (560x720) = 1280x720.

    The wide view is cropped to 560x720 (80px from each side).
    Renders at 640x640 max due to framebuffer limits, then upscales.
    """
    import cv2

    # Render wrist_cam at native 640x480, crop to 480x480, resize to 720x720
    wrist_renderer = mujoco.Renderer(env.model, height=480, width=640)
    wrist_renderer.update_scene(env.data, camera="wrist_cam")
    wrist_img = wrist_renderer.render()  # (480, 640, 3)
    # Center crop to 480x480
    crop_x = (640 - 480) // 2
    wrist_img = wrist_img[:, crop_x:crop_x + 480, :]
    # Resize to 720x720
    wrist_img = cv2.resize(wrist_img, (720, 720), interpolation=cv2.INTER_LINEAR)
    wrist_renderer.close()

    # Get wrist camera position for markers
    cam_world_pos = None
    cam_dir = None
    if show_cam_markers:
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        if cam_id >= 0:
            cam_world_pos = env.data.cam_xpos[cam_id].copy()
            cam_world_mat = env.data.cam_xmat[cam_id].reshape(3, 3)
            cam_dir = -cam_world_mat[:, 2]

    # Render wide view at 480x480 (max framebuffer height), then upscale
    wide_renderer = mujoco.Renderer(env.model, height=480, width=480)
    lookat, dist, azim, elev = ([0.25, -0.05, 0.05], 0.8, 135, -25)
    cam_obj = mujoco.MjvCamera()
    cam_obj.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_obj.lookat[:] = lookat
    cam_obj.distance = dist
    cam_obj.azimuth = azim
    cam_obj.elevation = elev
    wide_renderer.update_scene(env.data, camera=cam_obj)
    if cam_world_pos is not None and cam_dir is not None:
        add_camera_markers(wide_renderer._scene, cam_world_pos, cam_dir)
    wide_img = wide_renderer.render()  # (480, 480, 3)
    wide_renderer.close()

    # Upscale to 720x720
    wide_img = cv2.resize(wide_img, (720, 720), interpolation=cv2.INTER_LINEAR)

    # Crop wide to 560x720 (cut 80px from each side)
    wide_img = wide_img[:, 80:640, :]  # (720, 560, 3)

    # Combine side by side: 720 + 560 = 1280
    combined = np.concatenate([wrist_img, wide_img], axis=1)  # (720, 1280, 3)
    return combined


def evaluate_with_video(
    snapshot_path: Path,
    num_episodes: int = 5,
    output_dir: Path = None,
    cameras: list[str] = None,
    frame_size: int = 256,
    curriculum_stage: int = None,
    reward_version: str = None,
    save_video: bool = True,
    debug: bool = False,
    show_cam_markers: bool = False,
    x_format: bool = False,
):
    """Evaluate checkpoint and save multi-camera videos.

    Args:
        x_format: If True, render 16:9 format (1280x720) with wrist_cam + wide view.
    """

    if cameras is None:
        cameras = ["wrist_cam", "closeup", "wide", "wide2"]

    if output_dir is None:
        output_dir = snapshot_path.parent.parent / "eval_videos"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load snapshot to get config
    with open(snapshot_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", weights_only=False)

    # Try to get config from snapshot, otherwise load from run directory
    if "cfg" in payload:
        cfg = payload["cfg"]
    else:
        from src.training.config_loader import load_config
        config_path = snapshot_path.parent.parent / "config.yaml"
        if config_path.exists():
            print(f"Config not in snapshot, loading from {config_path}")
            cfg = load_config(str(config_path))
        else:
            raise FileNotFoundError(f"Config not found in snapshot or at {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use config values if not specified via args
    if curriculum_stage is None:
        curriculum_stage = cfg.env.get("curriculum_stage", 3)
    if reward_version is None:
        reward_version = cfg.env.get("reward_version", "v11")
    print(f"Using curriculum_stage={curriculum_stage}, reward_version={reward_version}")

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
            "gripper_state": [],
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
                gripper_state = base_env._get_gripper_state()
                debug_data["cube_z"].append(cube_z)
                debug_data["step_reward"].append(reward)
                debug_data["is_grasping"].append(is_grasping)
                debug_data["gripper_pos"].append(gripper_pos)
                debug_data["gripper_state"].append(gripper_state)

            # Render multi-camera view
            if save_video:
                base_env = env.unwrapped
                if x_format:
                    combined = render_x_format(base_env, show_cam_markers=show_cam_markers)
                else:
                    cam_frames = render_multi_camera(base_env, cameras, frame_size, show_cam_markers=show_cam_markers)
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
            gripper_state_arr = np.array(debug_data["gripper_state"])

            print(f"\n  --- Debug Summary ---")
            print(f"  Cube Z: min={cube_z_arr.min():.4f}, max={cube_z_arr.max():.4f}, final={cube_z_arr[-1]:.4f}")
            print(f"  Rewards: min={rewards_arr.min():.2f}, max={rewards_arr.max():.2f}, mean={rewards_arr.mean():.2f}")
            print(f"  Gripper: min={gripper_state_arr.min():.3f}, max={gripper_state_arr.max():.3f}, mean={gripper_state_arr.mean():.3f} (closed<0.25)")
            print(f"  Grasping: {grasping_arr.sum()}/{len(grasping_arr)} steps ({100*grasping_arr.mean():.1f}%)")

            # Find when cube first reaches threshold heights
            for threshold in [0.02, 0.04, 0.06, 0.08]:
                above = np.where(cube_z_arr > threshold)[0]
                if len(above) > 0:
                    print(f"  Cube > {threshold:.2f}m: first at step {above[0]}, {len(above)} total steps")

            # Save debug log to file
            debug_log_path = output_dir / f"episode_{ep:02d}_debug.txt"
            with open(debug_log_path, "w") as f:
                f.write("step,cube_z,reward,is_grasping,gripper_state,gripper_x,gripper_y,gripper_z\n")
                for i in range(len(debug_data["cube_z"])):
                    gp = debug_data["gripper_pos"][i]
                    gs = debug_data["gripper_state"][i]
                    f.write(f"{i},{debug_data['cube_z'][i]:.6f},{debug_data['step_reward'][i]:.4f},"
                           f"{int(debug_data['is_grasping'][i])},{gs:.4f},{gp[0]:.6f},{gp[1]:.6f},{gp[2]:.6f}\n")
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
    parser.add_argument("--cameras", nargs="+", default=["wrist_cam", "closeup", "wide", "wide2"],
                       help="Camera names to render")
    parser.add_argument("--frame_size", type=int, default=256, help="Frame size for rendering")
    parser.add_argument("--curriculum_stage", type=int, default=None, help="Curriculum stage (default: from config)")
    parser.add_argument("--reward_version", type=str, default=None, help="Reward version (default: from config)")
    parser.add_argument("--no_video", action="store_true", help="Skip video generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (cube_z, rewards, grasping)")
    parser.add_argument("--show-cam-markers", action="store_true", help="Show camera position markers (red sphere + green look direction)")
    parser.add_argument("--x-format", action="store_true", help="Render 16:9 format (1280x720) for X/Twitter")

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
        show_cam_markers=args.show_cam_markers,
        x_format=args.x_format,
    )


if __name__ == "__main__":
    main()
