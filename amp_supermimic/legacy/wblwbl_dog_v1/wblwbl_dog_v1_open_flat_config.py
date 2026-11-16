"""Pure-flat omnidirectional velocity-tracking task for WBLWBL Dog V1.

Inherits robot/AMP defaults from ``WblwblDogV1Cfg`` and enables:
- ``open_flat`` locked flat ground (no mountain/plateau curriculum);
- physical domain randomization (mass/inertia ±10%, kp/kd ±15%, effort ±10%,
  friction [0.3, 1.2], CoM offset, joint zero offset);
- adaptive velocity command curriculum (0.20 -> 2.00 m/s forward, etc.) driven by
  velocity/yaw tracking reward and fall rate;
- 10,000 iterations from scratch with stabilized actor learning rate (5e-5).
"""

from .wblwbl_dog_v1_config import WblwblDogV1Cfg, WblwblDogV1CfgTrain


class WblwblDogV1OpenFlatCfg(WblwblDogV1Cfg):
    class terrain(WblwblDogV1Cfg.terrain):
        profile = "open_flat"

    class commands(WblwblDogV1Cfg.commands):
        range_mode = "full"
        resample_steps = (100, 250)
        jump_height = 0.0
        jump_probability = 0.0

        class curriculum(WblwblDogV1Cfg.commands.curriculum):
            enabled = True
            update_interval_steps = 250
            initial_alpha = 0.0
            step_up = 0.10
            step_down = 0.05
            upgrade_vel_track = 0.70
            upgrade_yaw_track = 0.70
            upgrade_fall_rate = 0.10
            downgrade_vel_track = 0.45
            downgrade_fall_rate = 0.35
            fwd_vx_min = 0.20
            fwd_vx_start_max = 0.55
            fwd_vx_end_max = 2.00
            bwd_vx_min = 0.15
            bwd_vx_start_max = 0.45
            bwd_vx_end_max = 1.00
            strafe_vy_min = 0.10
            strafe_vy_start_max = 0.35
            strafe_vy_end_max = 0.75
            turn_wz_min = 0.30
            turn_wz_start_max = 0.90
            turn_wz_end_max = 1.60

    class domain_rand(WblwblDogV1Cfg.domain_rand):
        physical_domain_randomization = True
        self_collisions = False
        collision_mask_version = 2

    class amp(WblwblDogV1Cfg.amp):
        motion_config = "amp_supermimic/data/training/pool/train.json"
        amp_weight = 0.20
        foot_slide_weight = 0.008
        feet_close_weight = 0.008

    class rewards(WblwblDogV1Cfg.rewards):
        velocity_weight = 0.20
        velocity_sigma = 0.12
        yaw_weight = 0.09
        yaw_sigma = 0.10
        alive_weight = 0.02
        progress_trans_weight = 0.04
        progress_yaw_weight = 0.02
        dont_wait_trans_weight = -0.05
        dont_wait_yaw_weight = -0.04


class WblwblDogV1OpenFlatCfgTrain(WblwblDogV1CfgTrain):
    class algorithm(WblwblDogV1CfgTrain.algorithm):
        lr = 2e-4
        actor_lr = 5e-5
        gamma = 0.99
        lam = 0.95

    class world_model(WblwblDogV1CfgTrain.world_model):
        buffer_steps = 256
        utd = 0.4
        horizon = 32
        hist = 10
        wm_window = 8
        wm_batches = 8
        wm_batch = 2048
        pi_batch = 2048
        warmup_steps = 32
        wm_only_iters = 50

    class runner(WblwblDogV1CfgTrain.runner):
        experiment_name = "wblwbl_dog_v1"
        run_name = "open_flat_10k"
        run_dir = "amp_supermimic/outputs/trackerlab-amp/wblwbl_open_flat_10k"
        iterations = 10000
        log_every = 25
        ckpt_every = 500
        keep_every = 2000
