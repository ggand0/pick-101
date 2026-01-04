"""Config loading utilities that replace Hydra dependency.

Provides yaml-based config loading with OmegaConf for RoboBase compatibility.
"""

import importlib
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


# RoboBase defaults - extracted from robobase/cfgs/robobase_config.yaml
ROBOBASE_DEFAULTS = {
    "create_train_env": True,
    "num_train_envs": 1,
    "replay_size_before_train": 2000,
    "num_pretrain_steps": 0,
    "num_train_frames": 1000000,
    "eval_every_steps": 10000,
    "num_eval_episodes": 10,
    "update_every_steps": 2,
    "num_explore_steps": 2000,
    "save_snapshot": False,
    "snapshot_every_n": 1000,
    "batch_size": 256,
    "is_imitation_learning": False,
    "demos": 0,
    "demo_batch_size": None,
    "use_self_imitation": False,
    "pixels": False,
    "visual_observation_shape": [84, 84],
    "frame_stack": 1,
    "frame_stack_on_channel": True,
    "use_onehot_time_and_no_bootstrap": False,
    "action_repeat": 1,
    "action_sequence": 1,
    "execution_length": 1,
    "temporal_ensemble": True,
    "temporal_ensemble_gain": 0.01,
    "use_standardization": False,
    "use_min_max_normalization": False,
    "min_max_margin": 0.0,
    "norm_obs": False,
    "replay": {
        "prioritization": False,
        "size": 1000000,
        "gamma": 0.99,
        "demo_size": None,
        "save_dir": None,
        "nstep": 3,
        "num_workers": 4,
        "pin_memory": True,
        "alpha": 0.7,
        "beta": 0.5,
        "sequential": False,
        "transition_seq_len": 1,
    },
    "wandb": {
        "use": False,
        "project": "so101-rl",
        "entity": "gando",
        "name": "train",
    },
    "tb": {
        "use": True,
        "log_dir": "./runs/tb_logs",
        "name": "train",
    },
    "experiment_name": "exp",
    "seed": 1,
    "num_gpus": 1,
    "log_every": 1000,
    "log_train_video": False,
    "log_eval_video": True,
    "log_eval_video_every_n_evals": 1,  # Save video every N evals (1 = every eval)
    "log_pretrain_every": 100,
    "save_csv": False,
}

# DrQ-v2 method defaults - extracted from robobase/cfgs/method/drqv2.yaml
# Note: input_shapes, output_shape, num_envs are computed at runtime by DrQV2
DRQV2_DEFAULTS = {
    "method": {
        "_target_": "robobase.method.drqv2.DrQV2",
        "is_rl": True,
        "num_explore_steps": 2000,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "view_fusion_lr": 1e-4,
        "encoder_lr": 1e-4,
        "weight_decay": 0.0,
        "num_critics": 2,
        "critic_target_tau": 0.01,
        "stddev_schedule": "linear(1.0,0.1,500000)",
        "stddev_clip": 0.3,
        "use_augmentation": True,
        "actor_grad_clip": None,
        "critic_grad_clip": None,
        "bc_lambda": 0.0,
        "always_bootstrap": False,
        "action_sequence_network_type": "rnn",
        "critic_updates_shared_vision_encoder": True,
        "distributional_critic": False,
        "distributional_critic_limit": 20,
        "distributional_critic_atoms": 251,
        "distributional_critic_transform": True,
        "actor_model": {
            "_target_": "robobase.models.MLPWithBottleneckFeaturesAndSequenceOutput",
            "_partial_": True,
            # Runtime computed: input_shapes, output_shape, num_envs
            "num_rnn_layers": 1,
            "rnn_hidden_size": 128,
            "keys_to_bottleneck": ["fused_view_feats", "low_dim_obs", "time_obs"],
            "bottleneck_size": 50,
            "norm_after_bottleneck": True,
            "tanh_after_bottleneck": True,
            "mlp_nodes": [1024, 1024],
            "output_sequence_network_type": "rnn",
            "output_sequence_length": 1,  # Set by action_sequence
        },
        "critic_model": {
            "_target_": "robobase.models.MLPWithBottleneckFeatures",
            "_partial_": True,
            # Runtime computed: input_shapes, num_envs
            "output_shape": 1,  # Critic outputs single Q-value
            "num_rnn_layers": 1,
            "rnn_hidden_size": 128,
            "keys_to_bottleneck": ["fused_view_feats", "low_dim_obs", "time_obs"],
            "bottleneck_size": 50,
            "norm_after_bottleneck": True,
            "tanh_after_bottleneck": True,
            "mlp_nodes": [1024, 1024],
        },
        "encoder_model": {
            "_target_": "robobase.models.EncoderCNNMultiViewDownsampleWithStrides",
            "_partial_": True,
            # Runtime computed: input_shape
            "num_downsample_convs": 1,
            "num_post_downsample_convs": 3,
            "channels": 32,
            "kernel_size": 3,
        },
        "view_fusion_model": {
            "_target_": "robobase.models.FusionMultiCamFeature",
            "_partial_": True,
            # Runtime computed: input_shape
            "mode": "flatten",
        },
    },
}


def load_config(config_path: str) -> DictConfig:
    """Load yaml config and merge with RoboBase defaults.

    Args:
        config_path: Path to yaml config file

    Returns:
        DictConfig with all defaults merged
    """
    with open(config_path) as f:
        user_cfg = yaml.safe_load(f)

    # Start with RoboBase defaults
    cfg = OmegaConf.create(ROBOBASE_DEFAULTS)

    # Merge DrQ-v2 defaults
    cfg = OmegaConf.merge(cfg, OmegaConf.create(DRQV2_DEFAULTS))

    # Merge user config (overrides defaults)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(user_cfg))

    # Apply stddev_schedule from env to method if set
    if "env" in cfg and "stddev_schedule" in cfg.env:
        cfg.method.stddev_schedule = cfg.env.stddev_schedule

    # Apply num_explore_steps to method
    if "num_explore_steps" in cfg:
        cfg.method.num_explore_steps = cfg.num_explore_steps

    # Set action_sequence output length
    if "action_sequence" in cfg:
        cfg.method.actor_model.output_sequence_length = cfg.action_sequence

    return cfg


def instantiate(cfg: DictConfig, **kwargs) -> Any:
    """Instantiate object from config with _target_ field.

    Replacement for hydra.utils.instantiate(). Recursively instantiates
    nested configs that have _target_ fields.

    Args:
        cfg: Config dict with _target_ field
        **kwargs: Additional kwargs to pass to constructor

    Returns:
        Instantiated object
    """
    if not OmegaConf.is_dict(cfg):
        return cfg

    cfg = OmegaConf.to_container(cfg, resolve=True)

    if "_target_" not in cfg:
        return cfg

    target = cfg.pop("_target_")
    is_partial = cfg.pop("_partial_", False)

    # Recursively instantiate nested configs with _target_
    for key, value in list(cfg.items()):
        if isinstance(value, dict) and "_target_" in value:
            cfg[key] = _instantiate_dict(value)

    # Import the target class
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Merge cfg with kwargs
    cfg.update(kwargs)

    if is_partial:
        from functools import partial
        return partial(cls, **cfg)
    else:
        return cls(**cfg)


def _instantiate_dict(cfg: dict) -> Any:
    """Instantiate a plain dict config with _target_ field."""
    from functools import partial

    if "_target_" not in cfg:
        return cfg

    cfg = dict(cfg)  # Make a copy
    target = cfg.pop("_target_")
    is_partial = cfg.pop("_partial_", False)

    # Recursively instantiate nested configs
    for key, value in list(cfg.items()):
        if isinstance(value, dict) and "_target_" in value:
            cfg[key] = _instantiate_dict(value)

    # Import the target class
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if is_partial:
        return partial(cls, **cfg)
    else:
        return cls(**cfg)
