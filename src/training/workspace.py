"""Custom workspace with learning curve plotting."""

from pathlib import Path
import subprocess
import sys

from omegaconf import DictConfig

from robobase.workspace import Workspace
from robobase import utils


class SO101Workspace(Workspace):
    """Workspace with automatic learning curve plotting."""

    def __init__(self, cfg: DictConfig, env_factory):
        super().__init__(cfg, env_factory)
        self._plot_every_steps = 50000  # Plot every 50k steps
        self._last_plot_step = 0

    def _online_rl(self):
        """Override to add learning curve plotting."""
        train_until_frame = utils.Until(self.cfg.num_train_frames)
        seed_until_size = utils.Until(self.cfg.replay_size_before_train)
        should_log = utils.Every(self.cfg.log_every)
        eval_every_n = self.cfg.eval_every_steps if self.eval_env is not None else 0
        should_eval = utils.Every(eval_every_n)
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
                eval_metrics = self._eval()
                eval_metrics.update(self._get_common_metrics())
                self.logger.log_metrics(
                    eval_metrics, self.global_env_steps, prefix="eval"
                )
                if "episode_reward" in eval_metrics:
                    self.save_best_snapshot(eval_metrics["episode_reward"])

            if should_save_snapshot(self.main_loop_iterations):
                self.save_snapshot()

            # Plot learning curves every 50k steps
            if self.global_env_steps - self._last_plot_step >= self._plot_every_steps:
                self._plot_learning_curves()
                self._last_plot_step = self.global_env_steps

            self.logger.update_step(self.global_env_steps)

            if self._shutting_down:
                break

            self._main_loop_iterations += 1

        # Plot final learning curves at end of training
        self._plot_learning_curves(final=True)

    def _plot_learning_curves(self, final: bool = False):
        """Plot learning curves from tensorboard logs."""
        # Find tensorboard log directory
        tb_log_dir = None
        if hasattr(self.cfg, 'tb') and self.cfg.tb.use:
            tb_log_dir = Path(self.cfg.tb.log_dir) / self.cfg.tb.name

        if tb_log_dir is None or not tb_log_dir.exists():
            print(f"TB log dir not found: {tb_log_dir}")
            return

        # Output to run directory
        output_path = self.work_dir / "learning_curves.png"

        # Find the plotting script
        project_root = Path(__file__).parent.parent.parent
        plot_script = project_root / "scripts" / "plot_learning_curves.py"

        if not plot_script.exists():
            print(f"Plot script not found: {plot_script}")
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
