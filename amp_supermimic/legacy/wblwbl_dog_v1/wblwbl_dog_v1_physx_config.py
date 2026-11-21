"""Local Isaac Lab task defaults for WBLWBL command SuperMimic."""

from .wblwbl_dog_v1_config import WblwblDogV1Cfg, WblwblDogV1CfgTrain
from .wblwbl_dog_v1_open_flat_config import WblwblDogV1OpenFlatCfg, WblwblDogV1OpenFlatCfgTrain


class WblwblDogV1PhysxCfg(WblwblDogV1Cfg):
    class env(WblwblDogV1Cfg.env):
        num_envs = 256
        seed = 511
        solver_iterations = 8
        solver_velocity_iterations = 1
        action_limit = 4.0

    class robot(WblwblDogV1Cfg.robot):
        mjcf = None
        source_config = None
        urdf = "third_party/MyHIMLoco/legged_gym/resources/robots/wblwbl_dog_v1/urdf/wblwbl_dog.urdf"
        root_body = "base_link"
        legs = ("RR", "RL", "FR", "FL")
        standing_height = 0.40
        merge_fixed_joints = False
        default_q = (-.1, 1., -1.5, .1, 1., -1.5, -.1, .8, -1.5, .1, .8, -1.5)

    class control:
        physics_dt = .005
        decimation = 4
        action_scale = .25
        kp = 40.
        kd = 1.
        armature = .01
        project_joint_targets = False

    class terrain(WblwblDogV1Cfg.terrain):
        profile = "plane"
        spacing = 4.
        friction = 1.
        friction_range = (.3, 1.2)

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


class WblwblDogV1PhysxOpenFlatCfg(WblwblDogV1OpenFlatCfg):
    class env(WblwblDogV1OpenFlatCfg.env):
        # Keep PhysX's native single velocity iteration. Raising this to four
        # degraded direct reference replay and early training stability.
        solver_velocity_iterations = WblwblDogV1PhysxCfg.env.solver_velocity_iterations

    class robot(WblwblDogV1PhysxCfg.robot):
        # Retain explicit foot rigid bodies for PhysX contact handling. The
        # merged-foot variant lowered root height and did not improve replay.
        merge_fixed_joints = False

    control = WblwblDogV1PhysxCfg.control

    class terrain(WblwblDogV1PhysxCfg.terrain):
        profile = "open_flat"

    class commands(WblwblDogV1OpenFlatCfg.commands):
        samples = WblwblDogV1PhysxCfg.commands.samples


class WblwblDogV1PhysxOpenFlatCfgTrain(WblwblDogV1OpenFlatCfgTrain):
    class runner(WblwblDogV1OpenFlatCfgTrain.runner):
        run_name = "physx_open_flat_10k"
        run_dir = "amp_supermimic/outputs/physx/open-flat-10k"


class WblwblDogV1PhysxParkourCfg(WblwblDogV1PhysxOpenFlatCfg):
    class robot(WblwblDogV1PhysxOpenFlatCfg.robot):
        # Parkour contact sensors need the four named foot rigid bodies.
        merge_fixed_joints = False

    class terrain(WblwblDogV1PhysxOpenFlatCfg.terrain):
        profile = "parkour"
        level = None

    class commands(WblwblDogV1PhysxOpenFlatCfg.commands):
        curriculum = WblwblDogV1Cfg.commands.curriculum

    class amp(WblwblDogV1PhysxOpenFlatCfg.amp):
        reference_reset_probability = 0.


class WblwblDogV1PhysxParkourCfgTrain(WblwblDogV1PhysxOpenFlatCfgTrain):
    class runner(WblwblDogV1PhysxOpenFlatCfgTrain.runner):
        run_name = "physx_parkour"
        run_dir = "amp_supermimic/outputs/physx/parkour"



class WblwblDogV1PhysxBridgeCfg(WblwblDogV1PhysxWmpStairGapCfg):
    class env(WblwblDogV1PhysxCfg.env):
        action_limit = 20.
        num_envs = 512

    class terrain(WblwblDogV1PhysxWmpStairGapCfg.terrain):
        profile = "narrow_bridge"
        competition_stage = "learn"

    class commands(WblwblDogV1PhysxWmpStairGapCfg.commands):
        samples = ((.26, 0., 0.), (.32, 0., 0.), (.38, 0., 0.))

    class amp(WblwblDogV1PhysxCfg.amp):
        motion_config = "amp_supermimic/data/wblwbl/narrow_bridge_trot/train.json"
        amp_weight = .12

    class rewards(WblwblDogV1PhysxWmpStairGapCfg.rewards):
        lane_centering = .055
        foot_edges = .045
        strict_success = 2.


class WblwblDogV1PhysxBridgeCfgTrain(WblwblDogV1PhysxWmpStairGapCfgTrain):
    class world_model(WblwblDogV1PhysxCfgTrain.world_model):
        hist = 10

    class runner(WblwblDogV1PhysxWmpStairGapCfgTrain.runner):
        run_name = "bridge_trot"
        iterations = 6000


class WblwblDogV1PhysxWallCfg(WblwblDogV1PhysxBridgeCfg):
    class terrain(WblwblDogV1PhysxBridgeCfg.terrain):
        profile = "high_wall"

    class commands(WblwblDogV1PhysxBridgeCfg.commands):
        samples = ((.35, 0., 0.), (.45, 0., 0.), (.55, 0., 0.))

    class amp(WblwblDogV1PhysxBridgeCfg.amp):
        # A light approach prior. Geometry and real top contacts drive climb evaluation.
        motion_config = "amp_supermimic/data/flat_trainning/forward/a1_trot2/train.json"
        amp_weight = .025

    class rewards(WblwblDogV1PhysxBridgeCfg.rewards):
        top_support = 1.
        wall_clearance = .07
        wall_foot_placement = .065


class WblwblDogV1PhysxWallCfgTrain(WblwblDogV1PhysxBridgeCfgTrain):
    class runner(WblwblDogV1PhysxBridgeCfgTrain.runner):
        run_name = "high_wall"
