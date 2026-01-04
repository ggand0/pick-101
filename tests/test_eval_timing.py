"""Tests for eval and video saving timing logic."""

import pytest
import sys
sys.path.insert(0, "external/robobase")

from robobase.utils import Every, Until


class TestEvery:
    """Test the Every utility class used for eval timing."""

    def test_every_basic(self):
        """Every(10) triggers at 0, 10, 20, 30..."""
        every = Every(10)
        results = [every(i) for i in range(25)]
        expected = [i % 10 == 0 for i in range(25)]
        assert results == expected

    def test_every_zero_returns_true(self):
        """Every triggers at step 0."""
        every = Every(100)
        assert every(0) is True

    def test_every_none_returns_false(self):
        """Every(None) never triggers."""
        every = Every(None)
        assert every(0) is False
        assert every(100) is False

    def test_every_zero_interval_returns_false(self):
        """Every(0) never triggers."""
        every = Every(0)
        assert every(0) is False
        assert every(100) is False


class TestUntil:
    """Test the Until utility class used for training loop."""

    def test_until_basic(self):
        """Until(100) returns True while step < 100."""
        until = Until(100)
        assert until(0) is True
        assert until(50) is True
        assert until(99) is True
        assert until(100) is False
        assert until(150) is False

    def test_until_none_always_true(self):
        """Until(None) always returns True."""
        until = Until(None)
        assert until(0) is True
        assert until(1000000) is True


class TestEvalTiming:
    """Test eval timing with realistic config values."""

    def test_eval_every_100k_env_steps(self):
        """With 8 envs, action_repeat=2, eval_every_steps=6250 → eval every 100k env steps."""
        num_envs = 8
        action_repeat = 2
        eval_every_steps = 6250

        steps_per_iter = num_envs * action_repeat  # 16
        assert steps_per_iter == 16

        env_steps_per_eval = eval_every_steps * steps_per_iter
        assert env_steps_per_eval == 100_000

        # Verify eval triggers at correct iterations
        should_eval = Every(eval_every_steps)

        eval_iterations = []
        for i in range(100_000):  # Check first 100k iterations
            if should_eval(i):
                eval_iterations.append(i)

        # First 10 evals
        expected = [0, 6250, 12500, 18750, 25000, 31250, 37500, 43750, 50000, 56250]
        assert eval_iterations[:10] == expected

        # Convert to env steps
        eval_env_steps = [i * steps_per_iter for i in expected]
        expected_env_steps = [0, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000]
        assert eval_env_steps == expected_env_steps

    def test_resume_from_500k(self):
        """After resume from 500k, next eval is at 500k (immediately) then 600k, 700k..."""
        num_envs = 8
        action_repeat = 2
        eval_every_steps = 6250
        steps_per_iter = num_envs * action_repeat  # 16

        # Resume iteration
        resume_env_steps = 500_000
        resume_iter = resume_env_steps // steps_per_iter
        assert resume_iter == 31250

        # Check that eval triggers at resume
        should_eval = Every(eval_every_steps)
        assert should_eval(resume_iter) is True, "Eval should trigger immediately at resume"

        # Next evals after resume
        eval_iters_after_resume = []
        for i in range(resume_iter, resume_iter + 50000):
            if should_eval(i):
                eval_iters_after_resume.append(i)

        expected_iters = [31250, 37500, 43750, 50000, 56250, 62500, 68750, 75000]
        assert eval_iters_after_resume[:8] == expected_iters

        # Convert to env steps
        expected_env_steps = [500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000, 1_100_000, 1_200_000]
        actual_env_steps = [i * steps_per_iter for i in expected_iters]
        assert actual_env_steps == expected_env_steps


class TestVideoSaving:
    """Test video saving logic."""

    def test_video_every_eval(self):
        """With log_eval_video_every_n_evals=1, video saves at every eval."""
        video_every_n_evals = 1

        eval_count = 0
        videos_saved = []

        for eval_num in range(10):
            save_video = (eval_count % video_every_n_evals == 0)
            if save_video:
                videos_saved.append(eval_num)
            eval_count += 1

        # Video should save at every eval
        assert videos_saved == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_video_every_5_evals(self):
        """With log_eval_video_every_n_evals=5, video saves at eval 0, 5, 10..."""
        video_every_n_evals = 5

        eval_count = 0
        videos_saved = []

        for eval_num in range(15):
            save_video = (eval_count % video_every_n_evals == 0)
            if save_video:
                videos_saved.append(eval_num)
            eval_count += 1

        # Video should save at eval 0, 5, 10
        assert videos_saved == [0, 5, 10]

    def test_video_timing_after_resume(self):
        """After resume, eval_count resets to 0, so first eval saves video."""
        video_every_n_evals = 5

        # Simulate resume - eval_count resets to 0
        eval_count = 0

        # First eval after resume
        save_video = (eval_count % video_every_n_evals == 0)
        assert save_video is True, "First eval after resume should save video"


class TestFullTrainingLoop:
    """Integration test simulating the full training loop logic."""

    def test_training_500k_to_1500k_with_resume(self):
        """Simulate training from 500k to 1500k with resume."""
        # Config
        num_envs = 8
        action_repeat = 2
        eval_every_steps = 6250
        video_every_n_evals = 1
        num_train_frames = 1_500_000

        steps_per_iter = num_envs * action_repeat  # 16

        # Resume state
        main_loop_iterations = 31250  # 500k env steps
        eval_count = 0  # Resets on resume!

        # Utilities
        should_eval = Every(eval_every_steps)
        train_until = Until(num_train_frames)

        def global_env_steps():
            return main_loop_iterations * steps_per_iter

        # Track what happens
        evals_done = []
        videos_saved = []

        # Simulate training loop
        while train_until(global_env_steps()):
            if should_eval(main_loop_iterations):
                env_step = global_env_steps()
                evals_done.append(env_step)

                save_video = (eval_count % video_every_n_evals == 0)
                if save_video:
                    videos_saved.append(env_step)
                eval_count += 1

            main_loop_iterations += 1

        # Verify evals happened at 500k, 600k, 700k, ..., 1400k
        expected_evals = list(range(500_000, 1_500_000, 100_000))
        assert evals_done == expected_evals, f"Expected {expected_evals}, got {evals_done}"

        # With video_every_n_evals=1, all evals should have videos
        assert videos_saved == expected_evals, f"Expected videos at {expected_evals}, got {videos_saved}"

        # Verify we did 10 evals (500k, 600k, ..., 1400k)
        assert len(evals_done) == 10
        assert len(videos_saved) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
