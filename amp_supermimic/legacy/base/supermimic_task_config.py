"""Base task and training configs for SuperMimic command/AMP locomotion."""

from .base_config import BaseConfig


class SuperMimicRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        seed = 42
        device = "cuda:0"
        max_episode_steps = 1000
        solver_iterations = 50
        action_limit = 20.0
        clearance_ratio = 1.0

    class robot:
        name = "wblwbl_dog_v1"
        mjcf = None
        source_config = None

    class terrain:
        profile = "open_terrain"

    class commands:
        range_mode = "full"
        resample_steps = (100, 250)
        jump_height = 0.0
        jump_probability = 0.0

        class curriculum:
            enabled = False
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

    class domain_rand:
        # Enables upstream mass/inertia (±10%), kp/kd (±15%), effort (±10%) per world,
        # alongside base friction [0.3, 1.2], CoM offset (±0.025*scale), joint offset (±0.01 rad).
        physical_domain_randomization = False
        self_collisions = False
        collision_mask_version = 2

    class amp:
        motion_config = "amp_supermimic/data/flat_trainning/pool/train.json"
        amp_weight = 0.25
        amp_batch = 2048
        amp_updates = 2
        amp_history = 10
        style_feature_mode = "proprioceptive"
        reference_reset_probability = 0.7
        foot_slide_weight = 0.008
        feet_close_weight = 0.008

    class rewards:
        # Used by open_terrain / open_flat reward functions.
        velocity_weight = 0.20
        velocity_sigma = 0.12
        yaw_weight = 0.09
        yaw_sigma = 0.10
        alive_weight = 0.02
        upright_weight = -0.06
        clearance_weight = -0.04
        posture_weight = -0.0006
        action_rate_weight = -0.0008
        joint_vel_weight = -0.00002
        joint_limit_weight = -0.04
        progress_trans_weight = 0.04
        progress_yaw_weight = 0.02
        dont_wait_trans_weight = -0.05
        dont_wait_yaw_weight = -0.04


class SuperMimicCfgTrain(BaseConfig):
    class algorithm:
        lr = 2e-4
        actor_lr = 5e-5
        gamma = 0.99
        lam = 0.95

    class world_model:
        buffer_steps = 256
        utd = 0.4
        horizon = 32
        hist = 10
        wm_window = 8
        wm_batches = 8
        wm_batch = 2048
        pi_batch = 2048
        critic_epochs = 5
        critic_minibatches = 4
        warmup_steps = 32
        wm_only_iters = 50

    class runner:
        experiment_name = "supermimic"
        run_name = "default"
        run_dir = None
        iterations = 4000
        log_every = 25
        ckpt_every = 500
        keep_every = 2000
        speed_probe_updates = 1000
        compile = True
        compile_critic = True
        fused_optimizers = True
