"""WBLWBL Dog V1 parkour and stepping-stones configs for TrackerLab MuJoCo-Warp."""

from .wblwbl_dog_v1_config import WblwblDogV1Cfg, WblwblDogV1CfgTrain


class WblwblDogV1AmpSupermimicParkourCfg(WblwblDogV1Cfg):
    class env(WblwblDogV1Cfg.env):
        num_envs = 2048
        max_episode_steps = 1250  # 25 s at 50 Hz to traverse the full course.

    class terrain(WblwblDogV1Cfg.terrain):
        profile = "amp_supermimic_parkour"
        # Relative allocations of parallel environments (stepping stones excluded).
        terrain_proportions = {
            "flat_gap": 0.30,
            "tire_array": 0.30,
            "slope": 0.20,  # Split equally between smooth uphill and downhill lanes.
            "stair": 0.20,  # ROBOCON 3-riser up/platform/down stairs.
        }
        terrain_length = 8.0
        terrain_width = 2.4  # Lane pitch, including side trenches / separation.
        horizontal_scale = 0.02
        floor_height = -5.0
        spawn_x = -2.65  # Coordinates relative to the center of each course.
        finish_x = 3.80
        num_rows = 10  # Levels 0..9 match flat_gap, tire_array, and slope schedules.
        max_init_terrain_level = 0
        curriculum = True  # Success promotes, a fall demotes, and timeout holds the level.

        # Aligned 10-level curriculum: Level 8 matches ROBOCON 2027 spec, Level 9 adds ~10% margin.
        flat_gap_widths = (0.00, 0.045, 0.09, 0.135, 0.18, 0.225, 0.27, 0.315, 0.36, 0.42)
        flat_gap_corridor_width = (1.0, 2.0)
        flat_gap_platform_edges = (-1.0, 2.0)

        tire_heights = (0.03, 0.047, 0.064, 0.081, 0.098, 0.115, 0.132, 0.148, 0.165, 0.185)
        tire_corridor_width = 1.20
        tire_array_x = 2.00  # Spans [0.90, 3.10] m, leaving 0.70 m exit before finish_x=3.80.
        tire_pitch = 0.55
        tire_outer_radius = 0.275
        tire_inner_radius = 0.165
        tire_tread_depth_fraction = 0.04

        slope_grades = (0.00, 0.0455, 0.091, 0.1365, 0.182, 0.2275, 0.273, 0.3185, 0.364, 0.40)  # L8 = tan(20 deg).

        # ROBOCON 2027 沟壑组合 stairs: L8 = 3 risers of 120 mm (360 mm top), 320 mm depth, 800 mm width.
        stair_num_steps = 3
        stair_step_heights = (0.02, 0.0325, 0.045, 0.0575, 0.07, 0.0825, 0.095, 0.1075, 0.12, 0.13)
        stair_step_depths = (0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.30)
        stair_platform_length = 0.80
        stair_corridor_width = 0.80
        stair_center_x = 0.40

    class amp(WblwblDogV1Cfg.amp):
        motion_config = "amp_supermimic/data/parkour_training/pool/train.json"
        reference_reset_probability = 0.0

    class commands(WblwblDogV1Cfg.commands):
        resample_steps = (100, 150)  # 2.0 - 3.0 s at 50 Hz so yaw turns stay within the lane.
        vx_range = (0.25, 0.60)  # m/s
        wz_range = (-0.30, 0.30)  # rad/s
        wz_zero_probability = 0.30

    class rewards(WblwblDogV1Cfg.rewards):
        # Parkour task rewards managed via config inheritance and patched into CommandMDP.
        velocity_weight = 0.04
        velocity_sigma = 0.25
        yaw_weight = 0.04
        yaw_sigma = 0.25
        alive_weight = 0.06
        upright_weight = -0.06
        clearance_weight = -0.004
        posture_weight = -0.0006
        action_rate_weight = -0.0008
        joint_vel_weight = -0.00002
        joint_limit_weight = -0.04
        progress_weight = 0.008
        lane_weight = -0.001
        heading_weight = -0.003
        dont_wait_weight = -0.01
        success_bonus = 0.5
        fall_penalty = 0.5


class WblwblDogV1AmpSupermimicParkourCfgTrain(WblwblDogV1CfgTrain):
    class runner(WblwblDogV1CfgTrain.runner):
        run_name = "amp_supermimic_parkour"


# Backward-compatible aliases.
WblwblDogV1WmpParkourCfg = WblwblDogV1AmpSupermimicParkourCfg
WblwblDogV1WmpParkourCfgTrain = WblwblDogV1AmpSupermimicParkourCfgTrain
