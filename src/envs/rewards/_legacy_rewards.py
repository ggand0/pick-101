"""Legacy reward functions - kept for reference and checkpoint compatibility.

DO NOT USE FOR NEW TRAINING. These are historical experiments.
See lift_rewards.py for working reward functions (v11, v19).

Evolution:
- v1-v4: Early experiments with reach/grasp/lift combinations
- v5-v7: Push-down penalty experiments
- v8-v10: Curriculum learning additions (drop penalty, continuous lift)
- v12: Pick-and-place variant
- v13-v18: Image-based training iterations leading to v19
"""

import numpy as np


def reward_v1(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V1: Reach + grasp bonus + binary lift. Original reward that achieved grasping."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Grasp bonus (always)
    if info["is_grasping"]:
        reward += 0.25

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v2(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V2: Reach + continuous lift (no grasp condition). Disrupted grasping entirely."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Grasp bonus (stronger than V1)
    if info["is_grasping"]:
        reward += 0.5

    # Continuous lift (unconditional - this is what broke it)
    reward += max(0, (cube_z - 0.01) * 50.0)

    # Target height bonus
    if cube_z > env.lift_height:
        reward += 2.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v3(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V3: V1 + continuous lift gradient. Destabilized training."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Continuous lift baseline (without grasp)
    lift_baseline = max(0, (cube_z - 0.01) * 10.0)
    reward += lift_baseline

    # Grasp bonus (always)
    if info["is_grasping"]:
        reward += 0.25
        # Stronger lift reward when grasping
        reward += (cube_z - 0.01) * 40.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v4(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V4: V3 but grasp bonus only when elevated. Never closes gripper."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Continuous lift baseline
    lift_baseline = max(0, (cube_z - 0.01) * 10.0)
    reward += lift_baseline

    # Grasp bonus only when elevated
    if info["is_grasping"] and cube_z > 0.01:
        reward += 0.25
        reward += (cube_z - 0.01) * 40.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v5(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V5: V3 + push-down penalty. Nudge exploit - tilts cube."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Continuous lift baseline
    lift_baseline = max(0, (cube_z - 0.01) * 10.0)
    reward += lift_baseline

    # Grasp bonus (always)
    if info["is_grasping"]:
        reward += 0.25
        reward += (cube_z - 0.01) * 40.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v6(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V6: V5 without lift_baseline. Safe hover far away."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Grasp bonus (always)
    if info["is_grasping"]:
        reward += 0.25
        reward += (cube_z - 0.01) * 40.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v7(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V7: V1 + push-down penalty. Prevents agent from pushing cube into table."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Grasp bonus (always)
    if info["is_grasping"]:
        reward += 0.25

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v8(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V8: V7 + drop penalty. For curriculum learning with cube grasped at start."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty: penalize losing grasp after having it
    if was_grasping and not is_grasping:
        reward -= 2.0  # Significant penalty for dropping

    # Grasp bonus (always)
    if is_grasping:
        reward += 0.25

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v9(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V9: V8 + continuous lift gradient. For curriculum learning."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Reach reward (less important for curriculum where we start grasped)
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty: penalize losing grasp after having it
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping - this is the key addition
        # Reward proportional to height above table (0.015 is cube resting height)
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0  # Up to +2.0 at target height

    # Binary lift bonus (kept for compatibility)
    if cube_z > 0.02:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v10(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V10: v9 + target height bonus."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Target height bonus (the only addition from v9)
    if abs(cube_z - env.lift_height) < 0.005:
        reward += 1.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v12(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V12: Pick-and-place reward. Extends v11 with transport and placement rewards."""
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Phase 1: Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty (only during transport, not during intentional release)
    cube_to_target = info.get("cube_to_target", 0)
    if was_grasping and not is_grasping and cube_to_target > 0.03:
        reward -= 2.0  # Penalty for dropping away from target

    # Phase 2: Grasp and lift rewards
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Target height bonus
    if cube_z > env.lift_height:
        reward += 1.0

    # Phase 3: Transport reward (move cube toward target while lifted)
    if env._place_target_pos is not None:
        # Reward for cube being close to target (XY only)
        transport_reward = 1.0 - np.tanh(5.0 * cube_to_target)
        reward += transport_reward

        # Bonus for reaching target zone while grasping and lifted
        if cube_to_target < 0.03 and is_grasping and cube_z > env.lift_height:
            reward += 2.0  # At target, ready to place

        # Phase 4: Placement reward
        if cube_to_target < 0.03:
            # Reward for lowering cube at target
            if cube_z < env.lift_height:
                lower_progress = (env.lift_height - cube_z) / (env.lift_height - 0.015)
                reward += lower_progress * 1.0

            # Reward for releasing at target (gripper opening)
            if not is_grasping and cube_z < 0.025:
                reward += 3.0  # Just released at target

    # Action rate penalty for smoothness (only when lifted)
    if action is not None and cube_z > 0.06:
        action_delta = action - env._prev_action
        action_penalty = 0.01 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info.get("is_placed", False):
        reward += 10.0

    return reward


def reward_v13(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V13: v11 with binary lift bonus gated on is_grasping.

    Fixes exploit where agent tilts cube to get lift bonus without proper grasp.
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

        # Binary lift bonus (NOW GATED on is_grasping)
        if cube_z > 0.02:
            reward += 1.0

    # Target height bonus (aligned with success: z > lift_height)
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty for smoothness (only when lifted, to not hinder lifting)
    if action is not None and cube_z > 0.06:
        action_delta = action - env._prev_action
        action_penalty = 0.01 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v14(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V14: v13 with action penalty only during hold phase.

    Fixes the 7cm plateau issue where action penalty at 6cm+ blocked the final push to 8cm.
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    hold_count = info["hold_count"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

        # Binary lift bonus (gated on is_grasping)
        if cube_z > 0.02:
            reward += 1.0

    # Target height bonus (aligned with success: z > lift_height)
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty ONLY during hold phase at target height
    if action is not None and cube_z > env.lift_height and hold_count > 0:
        action_delta = action - env._prev_action
        action_penalty = 0.02 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v15(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V15: v14 + penalty for keeping gripper open too long."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    hold_count = info["hold_count"]
    gripper_state = info["gripper_state"]

    # Track consecutive steps with gripper open (more than initial 0.3)
    if gripper_state > 0.3:
        env._open_gripper_count += 1
    else:
        env._open_gripper_count = 0

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

        # Binary lift bonus (gated on is_grasping)
        if cube_z > 0.02:
            reward += 1.0

    # Target height bonus (aligned with success: z > lift_height)
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty ONLY during hold phase at target height
    if action is not None and cube_z > env.lift_height and hold_count > 0:
        action_delta = action - env._prev_action
        action_penalty = 0.02 * np.sum(action_delta**2)
        reward -= action_penalty

    # Gripper-open penalty: after 40 steps grace period, penalize keeping gripper open
    grace_period = 40
    if env._open_gripper_count > grace_period:
        excess_steps = env._open_gripper_count - grace_period
        open_penalty = min(0.05 * excess_steps / 50, 0.3)  # Grows over 50 steps, caps at 0.3
        reward -= open_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v16(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V16: v14 with increased grasp bonus (1.5 instead of 0.25)."""
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    hold_count = info["hold_count"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus (increased from 0.25 to 1.5)
    if is_grasping:
        reward += 1.5

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

        # Binary lift bonus (gated on is_grasping)
        if cube_z > 0.02:
            reward += 1.0

    # Target height bonus (aligned with success: z > lift_height)
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty ONLY during hold phase at target height
    if action is not None and cube_z > env.lift_height and hold_count > 0:
        action_delta = action - env._prev_action
        action_penalty = 0.02 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v17(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V17: Moving finger reach reward with contact cap.

    Addresses the exploration problem where agent never closes gripper because
    the standard gripper-to-cube reach gives no gradient for closing.
    """
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    gripper_state = info["gripper_state"]
    is_grasping = info["is_grasping"]
    hold_count = info["hold_count"]
    is_closed = gripper_state < 0.25

    # Standard gripper reach (static finger is part of gripper frame)
    gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_cube)

    # Moving finger reach - only applies when gripper is close to cube
    reach_threshold = 0.7  # ~3cm from cube
    if gripper_reach < reach_threshold:
        # Far from cube: just use gripper reach, no closing incentive yet
        reach_reward = gripper_reach
    else:
        # Close to cube: blend in moving finger reach
        if is_closed:
            moving_reach = 1.0  # Gripper closed, moving finger done its job
        else:
            moving_finger_pos = env._get_moving_finger_pos()
            moving_to_cube = np.linalg.norm(moving_finger_pos - cube_pos)
            moving_reach = 1.0 - np.tanh(10.0 * moving_to_cube)

        # Combined reach reward (0 to 1)
        reach_reward = (gripper_reach + moving_reach) * 0.5

    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus (1.5 from v16 - strong incentive to grasp)
    if is_grasping:
        reward += 1.5

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

        # Binary lift bonus (gated on is_grasping)
        if cube_z > 0.02:
            reward += 1.0

    # Target height bonus
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty during hold phase
    if action is not None and cube_z > env.lift_height and hold_count > 0:
        action_delta = action - env._prev_action
        action_penalty = 0.02 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v18(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V18: v17 with stronger lift incentive.

    Changes: Doubled lift coefficient (2.0 -> 4.0), added threshold ramp above 0.04m.
    """
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    gripper_state = info["gripper_state"]
    is_grasping = info["is_grasping"]
    hold_count = info["hold_count"]
    is_closed = gripper_state < 0.25

    # Standard gripper reach (static finger is part of gripper frame)
    gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_cube)

    # Moving finger reach - only applies when gripper is close to cube
    reach_threshold = 0.7  # ~3cm from cube
    if gripper_reach < reach_threshold:
        reach_reward = gripper_reach
    else:
        if is_closed:
            moving_reach = 1.0
        else:
            moving_finger_pos = env._get_moving_finger_pos()
            moving_to_cube = np.linalg.norm(moving_finger_pos - cube_pos)
            moving_reach = 1.0 - np.tanh(10.0 * moving_to_cube)

        reach_reward = (gripper_reach + moving_reach) * 0.5

    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 1.5

        # Continuous lift reward - DOUBLED from v17 (2.0 -> 4.0)
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 4.0

        # Binary lift bonus at 0.02m
        if cube_z > 0.02:
            reward += 1.0

        # Linear threshold ramp from 0.04m to 0.08m (NEW in v18)
        if cube_z > 0.04:
            threshold_progress = min(1.0, (cube_z - 0.04) / (env.lift_height - 0.04))
            reward += threshold_progress * 2.0

    # Target height bonus
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty during hold phase
    if action is not None and cube_z > env.lift_height and hold_count > 0:
        action_delta = action - env._prev_action
        action_penalty = 0.02 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward
