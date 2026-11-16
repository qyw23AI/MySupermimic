"""Base WBLWBL Dog V1 task and training configurations for SuperMimic."""

from ..base.supermimic_task_config import SuperMimicRobotCfg, SuperMimicCfgTrain


class WblwblDogV1Cfg(SuperMimicRobotCfg):
    class env(SuperMimicRobotCfg.env):
        num_envs = 4096
        seed = 42
        device = "cuda:0"
        max_episode_steps = 1000
        solver_iterations = 50
        action_limit = 20.0
        clearance_ratio = 1.0

    class robot(SuperMimicRobotCfg.robot):
        name = "wblwbl_dog_v1"
        mjcf = "../robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml"
        source_config = "him_supermimic/outputs/zero-start/b-seed511/config.json"

    class terrain(SuperMimicRobotCfg.terrain):
        profile = "open_terrain"

    class amp(SuperMimicRobotCfg.amp):
        motion_config = "amp_supermimic/data/training/pool/train.json"
        amp_weight = 0.25
        amp_batch = 2048
        amp_updates = 2
        amp_history = 10
        style_feature_mode = "proprioceptive"
        reference_reset_probability = 0.7
        foot_slide_weight = 0.008
        feet_close_weight = 0.008


class WblwblDogV1CfgTrain(SuperMimicCfgTrain):
    class runner(SuperMimicCfgTrain.runner):
        experiment_name = "wblwbl_dog_v1"
        run_name = "open_terrain"
        iterations = 4000
