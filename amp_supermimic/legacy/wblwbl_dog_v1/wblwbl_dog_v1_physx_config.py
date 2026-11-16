"""Local Isaac Lab task defaults for WBLWBL command SuperMimic."""

from .wblwbl_dog_v1_config import WblwblDogV1Cfg, WblwblDogV1CfgTrain


class WblwblDogV1PhysxCfg(WblwblDogV1Cfg):
    class env(WblwblDogV1Cfg.env):
        num_envs = 256
        seed = 511
        solver_iterations = 8
        action_limit = 4.0

    class robot(WblwblDogV1Cfg.robot):
        mjcf = None
        source_config = None
        urdf = "third_party/MyHIMLoco/legged_gym/resources/robots/wblwbl_dog_v1/urdf/wblwbl_dog.urdf"
        root_body = "base_link"
        legs = ("RR", "RL", "FR", "FL")
        standing_height = 0.40
        default_q = (-.1, 1., -1.5, .1, 1., -1.5, -.1, .8, -1.5, .1, .8, -1.5)

    class control:
        physics_dt = .005
        decimation = 4
        action_scale = .25
        kp = 40.
        kd = 1.
        armature = .01

    class terrain(WblwblDogV1Cfg.terrain):
        profile = "plane"
        spacing = 4.
        friction = 1.

    class commands(WblwblDogV1Cfg.commands):
        # Equiprobable forward, left and right command families.
        samples = ((.50, 0., 0.), (.20, 0., .60), (.20, 0., -.60))

    class rewards:
        # Multipliers for the shared TrackerLab continuous command terms.
        body_velocity = 1.
        yaw_rate = 1.

    class amp(WblwblDogV1Cfg.amp):
        motion_config = None
        motion_root = "amp_supermimic/data/wblwbl"
        clips = ("walk/ai4_dog_walk_03", "walk/vhdc_horse1_s1_walk_01",
                 "turn/atomic_turn_left", "turn/atomic_turn_right")
        heldout = ("walk/vhdc_horse1_s1_walk_02", "turn/ai4_dog_synth_circle_walk")
        amp_weight = .20
        amp_batch = 256
        reference_reset_probability = 0.


class WblwblDogV1PhysxCfgTrain(WblwblDogV1CfgTrain):
    class world_model(WblwblDogV1CfgTrain.world_model):
        wm_batch = 256
        pi_batch = 256

    class runner(WblwblDogV1CfgTrain.runner):
        iterations = 500
        run_name = "physx_flat"
        compile = False
        compile_critic = False
        fused_optimizers = False


class WblwblDogV1PhysxWmpStairGapCfg(WblwblDogV1PhysxCfg):
    class terrain(WblwblDogV1PhysxCfg.terrain):
        profile = "stair_gap"
        source = "../WMP/isaaclab_migration"
        level = 0
        competition_stage = "eval"
        competition_window = 50
        competition_promote = .80
        competition_demote = .40
        body_contact_limit = 1.

    class commands(WblwblDogV1PhysxCfg.commands):
        samples = ((.20, 0., 0.), (.25, 0., 0.), (.30, 0., 0.))

    class rewards(WblwblDogV1PhysxCfg.rewards):
        stair_support = .5
        landing = .5
        downhill = .5
        strict_success = 2.


class WblwblDogV1PhysxWmpStairGapCfgTrain(WblwblDogV1PhysxCfgTrain):
    class runner(WblwblDogV1PhysxCfgTrain.runner):
        run_name = "physx_wmp_stair_gap"
        ckpt_every = 300
        keep_every = 300


class WblwblDogV1PhysxWmpRampCfg(WblwblDogV1PhysxWmpStairGapCfg):
    class terrain(WblwblDogV1PhysxWmpStairGapCfg.terrain):
        profile = "stair_gap_ramp"


class WblwblDogV1PhysxWmpRampCfgTrain(WblwblDogV1PhysxWmpStairGapCfgTrain):
    class runner(WblwblDogV1PhysxWmpStairGapCfgTrain.runner):
        run_name = "physx_wmp_stair_gap_ramp"


class WblwblDogV1PhysxWmpCombosCfg(WblwblDogV1PhysxWmpStairGapCfg):
    class terrain(WblwblDogV1PhysxWmpStairGapCfg.terrain):
        profile = "gap_combos"
        competition_stage = "learn"


class WblwblDogV1PhysxWmpCombosCfgTrain(WblwblDogV1PhysxWmpStairGapCfgTrain):
    class runner(WblwblDogV1PhysxWmpStairGapCfgTrain.runner):
        run_name = "physx_wmp_gap_combos"
