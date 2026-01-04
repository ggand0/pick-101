"""Custom workspace for SO-101 training without Hydra dependency.

Based on robobase.workspace.Workspace but uses custom config loading.
"""

import logging
import random
import signal
import sys
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from robobase import utils
from robobase.envs.env import EnvFactory
from robobase.logger import Logger
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from robobase.video import VideoRecorder
import mujoco

from src.training.config_loader import instantiate


class MultiCameraVideoRecorder:
    """Video recorder that captures wrist_cam | closeup | wide views side-by-side."""

    def __init__(self, save_dir: Path, render_size: int = 256, fps: int = 20):
        self.save_dir = save_dir
        if save_dir is not None:
            self.save_dir.mkdir(exist_ok=True)
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self._renderer = None
        self._wrist_renderer = None  # Separate renderer for wrist cam (640x480)
        self._model = None

    def init(self, env, enabled: bool = True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if self.enabled:
            base_env = env.unwrapped
            self._model = base_env.model
            self._renderer = mujoco.Renderer(
                self._model, height=self.render_size, width=self.render_size
            )
            # Wrist cam renderer at real camera resolution (640x480)
            self._wrist_renderer = mujoco.Renderer(
                self._model, height=480, width=640
            )
        self.record(env)

    def _render_wrist_cam(self, data) -> np.ndarray:
        """Render wrist cam with real camera preprocessing (640x480 -> 480x480 center crop)."""
        import cv2

        self._wrist_renderer.update_scene(data, camera="wrist_cam")
        img = self._wrist_renderer.render()  # (480, 640, 3)

        # Center crop to 480x480 (crop 80px from each side)
        crop_x = (640 - 480) // 2  # 80
        img = img[:, crop_x:crop_x + 480, :]  # (480, 480, 3)

        # Resize to render_size for video
        img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_AREA)
        return img

    def _render_view(self, data, camera_config) -> np.ndarray:
        """Render a single view from camera config."""
        if isinstance(camera_config, str):
            self._renderer.update_scene(data, camera=camera_config)
        else:
            lookat, dist, azim, elev = camera_config
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = lookat
            cam.distance = dist
            cam.azimuth = azim
            cam.elevation = elev
            self._renderer.update_scene(data, camera=cam)
        return self._renderer.render().copy()

    def record(self, env):
        if not self.enabled:
            return

        base_env = env.unwrapped
        data = base_env.data

        # 3 views: wrist_cam | closeup | wide
        views = []

        # 1. Wrist cam (what agent sees - with real camera preprocessing)
        views.append(self._render_wrist_cam(data))

        # 2. Closeup (same as env.render() default - side view close to cube)
        # Original: lookat=[0.40, -0.10, 0.03], distance=0.35, azimuth=90, elevation=-15
        closeup_config = ([0.40, -0.10, 0.03], 0.35, 90, -15)
        views.append(self._render_view(data, closeup_config))

        # 3. Wide (diagonal view of arm and cube)
        # Original: lookat=[0.25, -0.05, 0.05], distance=0.8, azimuth=135, elevation=-25
        wide_config = ([0.25, -0.05, 0.05], 0.8, 135, -25)
        views.append(self._render_view(data, wide_config))

        # Combine horizontally
        combined = np.concatenate(views, axis=1)
        self.frames.append(combined)

    def save(self, file_name: str):
        if self.enabled and len(self.frames) > 0:
            import imageio
            path = self.save_dir / file_name
            imageio.mimsave(str(path), np.array(self.frames), fps=self.fps)

        # Cleanup renderers
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._wrist_renderer is not None:
            self._wrist_renderer.close()
            self._wrist_renderer = None

torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(int(seed))


def _create_default_replay_buffer(
    cfg: DictConfig,
    observation_space: gym.Space,
    action_space: gym.Space,
    demo_replay: bool = False,
) -> ReplayBuffer:
    extra_replay_elements = spaces.Dict({})
    if cfg.demos != 0:
        extra_replay_elements["demo"] = spaces.Box(0, 1, shape=(), dtype=np.uint8)
    replay_class = UniformReplayBuffer
    if cfg.replay.prioritization:
        replay_class = PrioritizedReplayBuffer
    replay_class = partial(
        replay_class,
        nstep=cfg.replay.nstep,
        gamma=cfg.replay.gamma,
    )
    return replay_class(
        save_dir=cfg.replay.save_dir,
        batch_size=cfg.batch_size if not demo_replay else cfg.demo_batch_size,
        replay_capacity=cfg.replay.size if not demo_replay else cfg.replay.demo_size,
        action_shape=action_space.shape,
        action_dtype=action_space.dtype,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=observation_space,
        extra_replay_elements=extra_replay_elements,
        num_workers=cfg.replay.num_workers,
        sequential=cfg.replay.sequential,
    )


class SO101Workspace:
    """Workspace for SO-101 training without Hydra dependency."""

    def __init__(
        self,
        cfg: DictConfig,
        env_factory: EnvFactory,
        work_dir: str,
        create_replay_fn: Callable[[DictConfig], ReplayBuffer] = None,
    ):
        if create_replay_fn is None:
            create_replay_fn = _create_default_replay_buffer

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        print(f"workspace: {self.work_dir}")

        # Sanity checks
        if (
            not cfg.is_imitation_learning
            and cfg.replay_size_before_train * cfg.action_repeat * cfg.action_sequence
            < cfg.env.episode_length
            and cfg.replay_size_before_train > 0
        ):
            raise ValueError(
                "replay_size_before_train * action_repeat "
                f"({cfg.replay_size_before_train} * {cfg.action_repeat}) "
                f"must be >= episode_length ({cfg.env.episode_length})."
            )

        if cfg.method.is_rl and cfg.action_sequence != 1:
            raise ValueError("Action sequence > 1 is not supported for RL methods")
        if cfg.method.is_rl and cfg.execution_length != 1:
            raise ValueError("execution_length > 1 is not supported for RL methods")
        if not cfg.method.is_rl and cfg.replay.nstep != 1:
            raise ValueError("replay.nstep != 1 is not supported for IL methods")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)

        # Device setup
        dev = "cpu"
        if cfg.num_gpus > 0:
            if sys.platform == "darwin":
                dev = "mps"
            else:
                dev = 0
        self.device = torch.device(dev)

        # Create logger
        self.logger = Logger(self.work_dir, cfg=self.cfg)
        self.env_factory = env_factory

        # Handle demos
        if (num_demos := cfg.demos) != 0:
            self.env_factory.collect_or_fetch_demos(cfg, num_demos)

        # Make training environment
        if cfg.num_train_envs > 0:
            self.train_envs = self.env_factory.make_train_env(cfg)
        else:
            self.train_envs = None
            logging.warning("Train env is not created. Training will not be supported")

        # Create evaluation environment
        self.eval_env = self.env_factory.make_eval_env(cfg)

        if num_demos != 0:
            self.env_factory.post_collect_or_fetch_demos(cfg)

        # Create the RL Agent using our custom instantiate
        observation_space = self.eval_env.observation_space
        action_space = self.eval_env.action_space

        intrinsic_reward_module = None
        if cfg.get("intrinsic_reward_module", None):
            intrinsic_reward_module = instantiate(
                cfg.intrinsic_reward_module,
                device=self.device,
                observation_space=observation_space,
                action_space=action_space,
            )

        self.agent = instantiate(
            cfg.method,
            device=self.device,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=cfg.num_train_envs,
            replay_alpha=cfg.replay.alpha,
            replay_beta=cfg.replay.beta,
            frame_stack_on_channel=cfg.frame_stack_on_channel,
            intrinsic_reward_module=intrinsic_reward_module,
        )
        self.agent.train(False)

        # Replay buffer
        self.replay_buffer = create_replay_fn(cfg, observation_space, action_space)
        self.prioritized_replay = cfg.replay.prioritization
        self.extra_replay_elements = self.replay_buffer.extra_replay_elements

        self.replay_loader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.batch_size,
            num_workers=cfg.replay.num_workers,
            pin_memory=cfg.replay.pin_memory,
            worker_init_fn=_worker_init_fn,
        )
        self._replay_iter = None

        # Demo replay buffer
        self.use_demo_replay = cfg.demo_batch_size is not None
        if self.use_demo_replay:
            self.demo_replay_buffer = create_replay_fn(
                cfg, observation_space, action_space, demo_replay=True
            )
            self.demo_replay_loader = DataLoader(
                self.demo_replay_buffer,
                batch_size=self.demo_replay_buffer.batch_size,
                num_workers=cfg.replay.num_workers,
                pin_memory=cfg.replay.pin_memory,
                worker_init_fn=_worker_init_fn,
            )

        # Video recorder (multi-camera: wrist_cam | closeup | wide)
        self.eval_video_recorder = MultiCameraVideoRecorder(
            (self.work_dir / "eval_videos") if self.cfg.log_eval_video else None
        )

        # State tracking
        self._timer = utils.Timer()
        self._pretrain_step = 0
        self._main_loop_iterations = 0
        self._global_env_episode = 0
        self._act_dim = self.eval_env.action_space.shape[0]
        if self.train_envs:
            self._episode_rollouts = [[] for _ in range(self.train_envs.num_envs)]
        else:
            self._episode_rollouts = []

        if cfg.num_eval_episodes == 0:
            self.eval_env.close()
            self.eval_env = None

        self._shutting_down = False
        self._best_eval_reward = float("-inf")

        # Learning curve plotting
        self._plot_every_steps = 50000
        self._last_plot_step = 0

    @property
    def pretrain_steps(self):
        return self._pretrain_step

    @property
    def main_loop_iterations(self):
        return self._main_loop_iterations

    @property
    def global_env_episodes(self):
        return self._global_env_episode

    @property
    def global_env_steps(self):
        if not self.train_envs:
            return 0
        return (
            self._main_loop_iterations
            * self.cfg.action_repeat
            * self.train_envs.num_envs
            * self.cfg.action_sequence
            + self.pretrain_steps
        )

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            _replay_iter = iter(self.replay_loader)
            if self.use_demo_replay:
                _demo_replay_iter = iter(self.demo_replay_loader)
                _replay_iter = utils.merge_replay_demo_iter(
                    _replay_iter, _demo_replay_iter
                )
            self._replay_iter = _replay_iter
        return self._replay_iter

    def train(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        if not self.train_envs:
            raise Exception("Train envs not created! Train can't be called!")
        try:
            self._train()
        except Exception as e:
            self.shutdown()
            raise e

    def _train(self):
        self._load_demos()
        self._pretrain_on_demos()
        self._online_rl()
        if self.cfg.save_snapshot:
            self.save_snapshot()
        self.shutdown()

    def _signal_handler(self, sig, frame):
        print("\nCtrl+C detected. Preparing to shutdown...")
        self._shutting_down = True

    def _load_demos(self):
        if (num_demos := self.cfg.demos) != 0:
            self.env_factory.load_demos_into_replay(
                self.cfg,
                self.replay_buffer,
                is_demo_buffer=True if self.cfg.is_imitation_learning else False,
            )
            if self.use_demo_replay:
                self.env_factory.load_demos_into_replay(
                    self.cfg, self.demo_replay_buffer, is_demo_buffer=True
                )

    def _pretrain_on_demos(self):
        if self.cfg.num_pretrain_steps > 0:
            pre_train_until_step = utils.Until(self.cfg.num_pretrain_steps)
            should_pretrain_log = utils.Every(self.cfg.log_pretrain_every)
            should_pretrain_eval = utils.Every(self.cfg.eval_every_steps)

            while pre_train_until_step(self.pretrain_steps):
                self.agent.logging = should_pretrain_log(self.pretrain_steps)
                pretrain_metrics = self._perform_updates()

                if self.agent.logging:
                    pretrain_metrics.update(self._get_common_metrics())
                    self.logger.log_metrics(
                        pretrain_metrics, self.pretrain_steps, prefix="pretrain"
                    )

                if should_pretrain_eval(self.pretrain_steps):
                    eval_metrics = self._eval()
                    eval_metrics.update(self._get_common_metrics())
                    self.logger.log_metrics(
                        eval_metrics, self.pretrain_steps, prefix="pretrain_eval"
                    )

                self._pretrain_step += 1

    def _perform_updates(self) -> dict[str, Any]:
        if self.agent.logging:
            start_time = time.time()
        metrics = {}
        self.agent.train(True)
        for i in range(self.train_envs.num_envs):
            if (self.main_loop_iterations + i) % self.cfg.update_every_steps != 0:
                continue
            metrics.update(
                self.agent.update(
                    self.replay_iter, self.main_loop_iterations + i, self.replay_buffer
                )
            )
        self.agent.train(False)
        if self.agent.logging:
            execution_time = time.time() - start_time
            metrics["agent_batched_updates_per_second"] = (
                self.train_envs.num_envs / execution_time
            )
        return metrics

    def _perform_env_steps(
        self, observations: dict[str, np.ndarray], env: gym.Env, eval_mode: bool
    ) -> tuple[np.ndarray, tuple, dict[str, Any]]:
        if self.agent.logging:
            start_time = time.time()

        with torch.no_grad(), utils.eval_mode(self.agent):
            torch_observations = {
                k: torch.from_numpy(v).to(self.device) for k, v in observations.items()
            }
            if eval_mode:
                torch_observations = {
                    k: v.unsqueeze(0) for k, v in torch_observations.items()
                }
            action = self.agent.act(
                torch_observations, self.main_loop_iterations, eval_mode=eval_mode
            )
            metrics = {}
            if isinstance(action, tuple):
                action, act_info = action
                metrics["agent_act_info"] = act_info
            action = action.cpu().detach().numpy()
            if action.ndim != 3:
                raise ValueError(
                    "Expected actions from `agent.act` to have shape "
                    "(Batch, Timesteps, Action Dim)."
                )
            if eval_mode:
                action = action[0]  # batch of 1 for eval

        if self.agent.logging:
            execution_time = time.time() - start_time
            metrics["agent_act_steps_per_second"] = (
                self.train_envs.num_envs / execution_time
            )
            start_time = time.time()

        *env_step_tuple, next_info = env.step(action)

        if self.agent.logging:
            execution_time = time.time() - start_time
            metrics["env_steps_per_second"] = (
                self.train_envs.num_envs / execution_time
            )
            for k, v in next_info.items():
                metrics[f"env_info/{k}"] = v if eval_mode else v[0]

        return action, (*env_step_tuple, next_info), metrics

    def _add_to_replay(
        self,
        actions,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
        next_infos,
    ):
        # Convert observation to list of observations ordered by train_env index
        list_of_obs_dicts = [
            dict(zip(observations, t)) for t in zip(*observations.values())
        ]
        agents_reset = []

        for i in range(self.train_envs.num_envs):
            # Add transitions to episode rollout
            self._episode_rollouts[i].append(
                (
                    actions[i],
                    list_of_obs_dicts[i],
                    rewards[i],
                    terminations[i],
                    truncations[i],
                    {k: infos[k][i] for k in infos.keys()},
                    {k: next_infos[k][i] for k in next_infos.keys()},
                )
            )

            # If episode finishes, add to replay buffer
            if terminations[i] or truncations[i]:
                agents_reset.append(i)
                ep = self._episode_rollouts[i]
                last_next_info = ep[-1][-1]
                assert last_next_info["_final_observation"]
                final_obs = last_next_info["final_observation"]
                final_info = last_next_info["final_info"]
                task_success = int(final_info.get("task_success", 0) > 0.0)

                relabeling_as_demo = (
                    task_success
                    and self.use_demo_replay
                    and self.cfg.use_self_imitation
                )

                for act, obs, rew, term, trunc, info, next_info in ep:
                    # Only keep the last frames regardless of frame stacks
                    obs = {k: v[-1] for k, v in obs.items()}
                    # Strip out temporal dimension as action_sequence = 1
                    act = act[0]

                    if relabeling_as_demo:
                        info["demo"] = 1
                    else:
                        info["demo"] = 0

                    # Filter out unwanted keys in info
                    extra_replay_elements = {
                        k: v
                        for k, v in info.items()
                        if k in self.extra_replay_elements.keys()
                    }

                    self.replay_buffer.add(
                        obs, act, rew, term, trunc, **extra_replay_elements
                    )
                    if relabeling_as_demo:
                        self.demo_replay_buffer.add(
                            obs, act, rew, term, trunc, **extra_replay_elements
                        )

                # Add final obs
                final_obs = {k: v[-1] for k, v in final_obs.items()}
                self.replay_buffer.add_final(final_obs)
                if relabeling_as_demo:
                    self.demo_replay_buffer.add_final(final_obs)

                # Clean up
                self._global_env_episode += 1
                self._episode_rollouts[i].clear()

        self.agent.reset(self.main_loop_iterations, agents_reset)

    def _eval(self, eval_record_all_episode: bool = False, save_video_and_log: bool = False) -> dict[str, Any]:
        self.agent.set_eval_env_running(True)
        step, episode, total_reward, successes = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        first_rollout = []
        metrics = {}

        # Debug data collection for logging (when saving video)
        all_episode_data = []

        while eval_until_episode(episode):
            observation, info = self.eval_env.reset()
            self.agent.reset(self.main_loop_iterations, [self.train_envs.num_envs])
            enabled = save_video_and_log and (eval_record_all_episode or episode == 0)
            self.eval_video_recorder.init(self.eval_env, enabled=enabled)
            termination, truncation = False, False

            # Per-episode debug data
            episode_data = {
                "cube_z": [], "step_reward": [], "is_grasping": [],
                "gripper_pos": [], "gripper_state": [],
            }

            while not (termination or truncation):
                (
                    action,
                    (next_observation, reward, termination, truncation, next_info),
                    env_metrics,
                ) = self._perform_env_steps(observation, self.eval_env, True)
                observation = next_observation
                info = next_info
                metrics.update(env_metrics)
                if "agent_act_info" in env_metrics:
                    if hasattr(self.eval_env, "give_agent_info"):
                        self.eval_env.give_agent_info(env_metrics["agent_act_info"])
                self.eval_video_recorder.record(self.eval_env)
                total_reward += reward
                step += 1

                # Collect debug data if saving
                if save_video_and_log:
                    base_env = self.eval_env.unwrapped
                    episode_data["cube_z"].append(info.get("cube_z", 0))
                    episode_data["step_reward"].append(reward)
                    episode_data["is_grasping"].append(info.get("is_grasping", False))
                    episode_data["gripper_pos"].append(info.get("gripper_pos", [0, 0, 0]))
                    if hasattr(base_env, "_get_gripper_state"):
                        episode_data["gripper_state"].append(base_env._get_gripper_state())

            if episode == 0:
                first_rollout = np.array(self.eval_video_recorder.frames)
            if save_video_and_log:
                self.eval_video_recorder.save(f"{self.global_env_steps}.mp4")
                all_episode_data.append(episode_data)

            success = info.get("task_success")
            if success is not None:
                successes += np.array(success).astype(int).item()
            else:
                successes = None
            episode += 1

        # Save debug log alongside video
        if save_video_and_log and all_episode_data:
            self._save_eval_debug_log(all_episode_data)

        metrics.update(
            {
                "episode_reward": total_reward / episode,
                "episode_length": step * self.cfg.action_repeat / episode,
            }
        )
        if successes is not None:
            metrics["episode_success"] = successes / episode
        if self.cfg.log_eval_video and len(first_rollout) > 0:
            metrics["eval_rollout"] = dict(video=first_rollout, fps=4)
        self.agent.set_eval_env_running(False)
        return metrics

    def _save_eval_debug_log(self, all_episode_data: list[dict]):
        """Save debug log text file alongside eval video."""
        log_path = self.work_dir / "eval_videos" / f"{self.global_env_steps}_debug.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as f:
            f.write(f"Eval Debug Log @ step {self.global_env_steps}\n")
            f.write("=" * 60 + "\n\n")

            for ep_idx, ep_data in enumerate(all_episode_data):
                cube_z = np.array(ep_data["cube_z"])
                rewards = np.array(ep_data["step_reward"])
                grasping = np.array(ep_data["is_grasping"])
                gripper_state = np.array(ep_data["gripper_state"]) if ep_data["gripper_state"] else None

                f.write(f"Episode {ep_idx}:\n")
                f.write(f"  Steps: {len(cube_z)}\n")
                f.write(f"  Total reward: {rewards.sum():.3f}\n")
                f.write(f"  Cube Z: min={cube_z.min():.4f}, max={cube_z.max():.4f}, final={cube_z[-1]:.4f}\n")
                f.write(f"  Grasping: {grasping.sum()} steps ({100*grasping.mean():.1f}%)\n")
                if gripper_state is not None and len(gripper_state) > 0:
                    f.write(f"  Gripper: min={gripper_state.min():.3f}, max={gripper_state.max():.3f}, mean={gripper_state.mean():.3f} (closed<0.25)\n")
                f.write("\n")

                # CSV-style per-step data
                f.write("  step,cube_z,reward,is_grasping,gripper_state\n")
                for i in range(len(cube_z)):
                    gs = gripper_state[i] if gripper_state is not None and i < len(gripper_state) else 0
                    f.write(f"  {i},{cube_z[i]:.4f},{rewards[i]:.4f},{int(grasping[i])},{gs:.4f}\n")
                f.write("\n")

        print(f"Saved eval debug log to {log_path}")

    def _online_rl(self):
        train_until_frame = utils.Until(self.cfg.num_train_frames)
        seed_until_size = utils.Until(self.cfg.replay_size_before_train)
        should_log = utils.Every(self.cfg.log_every)
        eval_every_n = self.cfg.eval_every_steps if self.eval_env is not None else 0
        should_eval = utils.Every(eval_every_n)
        # Video saving: every N evals (not env steps - that design was broken)
        video_every_n_evals = getattr(self.cfg, "log_eval_video_every_n_evals", 1)
        eval_count = 0
        snapshot_every_n = self.cfg.snapshot_every_n if self.cfg.save_snapshot else 0
        should_save_snapshot = utils.Every(snapshot_every_n)

        observations, info = self.train_envs.reset()
        agent_0_ep_len = agent_0_reward = 0
        agent_0_prev_ep_len = agent_0_prev_reward = agent_0_prev_success = None

        while train_until_frame(self.global_env_steps):
            metrics = {}
            self.agent.logging = False
            if should_log(self.main_loop_iterations):
                self.agent.logging = True
            if not seed_until_size(len(self.replay_buffer)):
                update_metrics = self._perform_updates()
                metrics.update(update_metrics)

            (
                action,
                (next_observations, rewards, terminations, truncations, next_info),
                env_metrics,
            ) = self._perform_env_steps(observations, self.train_envs, False)

            agent_0_reward += rewards[0]
            agent_0_ep_len += 1
            if terminations[0] or truncations[0]:
                agent_0_prev_ep_len = agent_0_ep_len
                agent_0_prev_reward = agent_0_reward
                final_info = next_info.get("final_info", [{}])[0] if "final_info" in next_info else {}
                agent_0_prev_success = float(final_info.get("task_success", 0) > 0)
                agent_0_ep_len = agent_0_reward = 0

            metrics.update(env_metrics)
            self._add_to_replay(
                action,
                observations,
                rewards,
                terminations,
                truncations,
                info,
                next_info,
            )
            observations = next_observations
            info = next_info

            if should_log(self.main_loop_iterations):
                metrics.update(self._get_common_metrics())
                if agent_0_prev_reward is not None and agent_0_prev_ep_len is not None:
                    metrics.update(
                        {
                            "episode_reward": agent_0_prev_reward,
                            "episode_length": agent_0_prev_ep_len * self.cfg.action_repeat,
                        }
                    )
                    if agent_0_prev_success is not None:
                        metrics["episode_success"] = agent_0_prev_success
                self.logger.log_metrics(metrics, self.global_env_steps, prefix="train")

            if should_eval(self.main_loop_iterations):
                # Save video every N evals (simpler than trying to align env steps)
                save_video = (eval_count % video_every_n_evals == 0) and self.cfg.log_eval_video
                eval_count += 1
                eval_metrics = self._eval(save_video_and_log=save_video)
                eval_metrics.update(self._get_common_metrics())
                self.logger.log_metrics(
                    eval_metrics, self.global_env_steps, prefix="eval"
                )
                if "episode_reward" in eval_metrics:
                    self.save_best_snapshot(eval_metrics["episode_reward"])

            if should_save_snapshot(self.main_loop_iterations):
                self.save_snapshot()

            # Plot learning curves periodically
            if self.global_env_steps - self._last_plot_step >= self._plot_every_steps:
                self._plot_learning_curves()
                self._last_plot_step = self.global_env_steps

            self.logger.update_step(self.global_env_steps)

            if self._shutting_down:
                break

            self._main_loop_iterations += 1

        # Plot final learning curves
        self._plot_learning_curves(final=True)

    def _get_common_metrics(self) -> dict[str, Any]:
        _, total_time = self._timer.reset()
        metrics = {
            "total_time": total_time,
            "iteration": self.main_loop_iterations,
            "env_steps": self.global_env_steps,
            "env_episodes": self.global_env_episodes,
            "buffer_size": len(self.replay_buffer),
        }
        if self.use_demo_replay:
            metrics["demo_buffer_size"] = len(self.demo_replay_buffer)
        return metrics

    def shutdown(self):
        if self.eval_env:
            self.eval_env.close()
        if self.train_envs:
            self.train_envs.close()
        self.replay_buffer.shutdown()
        if self.use_demo_replay:
            self.demo_replay_buffer.shutdown()

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshots" / f"{self.global_env_steps}_snapshot.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        keys_to_save = [
            "_pretrain_step",
            "_main_loop_iterations",
            "_global_env_episode",
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload["agent"] = self.agent.state_dict()
        payload["cfg"] = self.cfg
        torch.save(payload, snapshot)

        # Also save as latest
        latest = self.work_dir / "snapshots" / "latest_snapshot.pt"
        torch.save(payload, latest)
        print(f"Saved snapshot to {snapshot}")

    def save_best_snapshot(self, eval_reward: float):
        if eval_reward > self._best_eval_reward:
            self._best_eval_reward = eval_reward
            best_path = self.work_dir / "snapshots" / "best_snapshot.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            keys_to_save = [
                "_pretrain_step",
                "_main_loop_iterations",
                "_global_env_episode",
            ]
            payload = {k: self.__dict__[k] for k in keys_to_save}
            payload["agent"] = self.agent.state_dict()
            payload["cfg"] = self.cfg
            payload["best_eval_reward"] = eval_reward
            torch.save(payload, best_path)
            print(f"New best eval reward: {eval_reward:.2f}, saved to {best_path}")

    def load_snapshot(self, snapshot_path: Path):
        snapshot = torch.load(snapshot_path, map_location=self.device, weights_only=False)
        for k, v in snapshot.items():
            if k == "agent":
                self.agent.load_state_dict(v)
            elif k == "best_eval_reward":
                self._best_eval_reward = v
            elif k == "cfg":
                # Don't overwrite current cfg - allows resuming with different num_train_frames
                pass
            else:
                self.__dict__[k] = v
        print(f"Loaded snapshot from {snapshot_path}, iteration {self._main_loop_iterations}")

    def _plot_learning_curves(self, final: bool = False):
        # Tensorboard logs are now saved per-run in work_dir/tb_logs
        tb_log_dir = self.work_dir / "tb_logs"

        if not tb_log_dir.exists():
            return

        output_path = self.work_dir / "learning_curves.png"
        project_root = Path(__file__).parent.parent.parent
        plot_script = project_root / "scripts" / "plot_learning_curves.py"

        if not plot_script.exists():
            return

        try:
            status = "final" if final else f"{self.global_env_steps // 1000}k"
            print(f"\n[{status}] Plotting learning curves...")

            result = subprocess.run(
                [
                    sys.executable,
                    str(plot_script),
                    "--log_dir", str(tb_log_dir),
                    "--output", str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                print(f"Learning curves saved to {output_path}")
            else:
                print(f"Plot failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("Plot script timed out")
        except Exception as e:
            print(f"Failed to plot learning curves: {e}")
