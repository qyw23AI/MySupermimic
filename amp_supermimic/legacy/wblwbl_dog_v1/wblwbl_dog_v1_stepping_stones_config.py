"""Dedicated ROBOCON 2027 stepping-stones (梅花桩) task config for WBLWBL Dog V1.

Inherits parkour base defaults from ``WblwblDogV1AmpSupermimicParkourCfg`` and
isolates all stepping-stones terrain, command, AMP, and special foothold/pit/
completion rewards for independent tuning.
"""

from .wblwbl_dog_v1_wmp_parkour_config import (
    WblwblDogV1AmpSupermimicParkourCfg,
    WblwblDogV1AmpSupermimicParkourCfgTrain,
)


class WblwblDogV1AmpSupermimicSteppingStonesCfg(WblwblDogV1AmpSupermimicParkourCfg):
    class env(WblwblDogV1AmpSupermimicParkourCfg.env):
        num_envs = 2048
        max_episode_steps = 1000  # 20 s at 50 Hz for the 3.5 m stepping-stones course.

    class terrain(WblwblDogV1AmpSupermimicParkourCfg.terrain):
        profile = "amp_supermimic_stepping_stones"
        terrain_proportions = {
            "stepping_stones": 1.0,
        }
        spawn_x = -2.35  # 0.60 m flat run-up before the 0.80 m entry platform (at L7).
        finish_x = 2.65  # 0.90 m exit beyond the 0.80 m landing platform (at L7).
        num_rows = 15

        # ROBOCON 2027 (rev. 2026-10-29): 200x200x100 mm blocks, 800x800x100 mm platforms.
        plum_stone_size = 0.20
        plum_stone_height = 0.10
        plum_platform_size = 0.80
        plum_road_width = 0.80
        # Level 7 matches ROBOCON standard (400 mm same-row gap, 200 mm row gap, 100 mm platform gap).
        plum_longitudinal_gaps = (-0.075, -0.050, -0.025, 0.000, 0.025, 0.050, 0.075, 0.100,
                                  0.1125, 0.125, 0.1375, 0.150, 0.1625, 0.175, 0.1875)
        plum_row_gaps = (0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200,
                         0.2125, 0.225, 0.2375, 0.250, 0.2625, 0.275, 0.2875)
        plum_platform_gaps = (0.0125, 0.025, 0.0375, 0.050, 0.0625, 0.075, 0.0875, 0.100,
                              0.1125, 0.125, 0.1375, 0.150, 0.1625, 0.175, 0.1875)

    class commands(WblwblDogV1AmpSupermimicParkourCfg.commands):
        resample_steps = (100, 150)
        vx_range = (0.20, 0.50)  # m/s, steadier pace for 20 cm stepping stones.
        wz_range = (-0.20, 0.20)  # rad/s, gentle heading corrections without stepping off stones.
        wz_zero_probability = 0.35

    class amp(WblwblDogV1AmpSupermimicParkourCfg.amp):
        motion_config = "amp_supermimic/data/parkour_training/pool/train.json"
        amp_weight = 0.20  # Balanced so AMP shapes natural gait without overpowering foothold precision.
        reference_reset_probability = 0.0

    class rewards(WblwblDogV1AmpSupermimicParkourCfg.rewards):
        # --- Inherited base command & regularization terms (per 20 ms step) ---
        velocity_weight = 0.03
        yaw_weight = 0.03
        alive_weight = 0.05
        upright_weight = -0.06
        clearance_weight = -0.002  # Relaxed from -0.004 to allow natural height adjustment over 10 cm blocks.
        progress_weight = 0.008
        lane_weight = -0.002
        heading_weight = -0.005
        dont_wait_weight = -0.015

        # --- Terminal payouts (gamma = 0.99; continuation is added automatically to success_bonus) ---
        success_bonus = 2.0
        fall_penalty = 1.0

        # --- Stepping-stones special terrain rewards (differentiable, used in imagination & physics) ---
        stone_foothold_weight = 0.050  # Full-sole interior support on stones/platforms (margin check).
        stone_edge_margin = 0.035  # Sample radius around foot center (ROBOCON max foot radius <= 40 mm).
        stone_touch_progress_weight = 0.040  # Touching the 6 stones while moving forward (anti-camping).
        stone_pit_contact_weight = -0.12  # Penalty when feet contact z=0 floor inside the stone zone.
        stone_pit_fail_feet = 2  # ROBOCON rule: >= 2 feet touching floor fails & terminates immediately.
        stone_swing_clearance_weight = 0.020  # Swing foot clearance above current & ahead terrain.
        stone_swing_target_clearance = 0.08  # Target swing clearance in meters.
        stone_slip_weight = -0.015  # Penalize horizontal foot slip while supported.
        stone_zone_margin = 0.02  # Edge inset from platforms before pit-contact rule activates.

        # --- Episode-level per-stone coverage & curriculum gate (tracked in physical CommandSim) ---
        stone_visit_bonus = 1.5  # One-time bonus per newly contacted stone (6 stones -> +9.0 total).
        stone_all_visited_goal_bonus = 10.0  # Bonus when all 6 stones were contacted AND finish_x reached.
        stone_require_all_for_curriculum = True  # Only promote terrain level when all 6 stones are touched.


class WblwblDogV1AmpSupermimicSteppingStonesCfgTrain(WblwblDogV1AmpSupermimicParkourCfgTrain):
    class runner(WblwblDogV1AmpSupermimicParkourCfgTrain.runner):
        run_name = "amp_supermimic_stepping_stones"
