# 来源与固定代码位置

所有上游链接固定到本次子模块提交。本机链接对应审计时的工作树；完整哈希见 manifest。

- **O-POL** — [Official policy learners](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/policy_learner.py)。位置：BpttPolicyLearner.update/_train_batch; ActorCriticPolicyLearner._bootstrap/_fit_critic; gae_returns。
- **O-WM** — [Official supervised world-model learner](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/wm_learner.py)。位置：WorldModelLearnerCfg; _rollout; _compute_losses; update_normalization。
- **O-RUN** — [Official runner](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/supermimic_runner.py)。位置：BaseRunnerCfg; build_world_model; _step_and_store; SuperMimicRunner.run。
- **O-ENV** — [Official differentiable world-model environment](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/envs/world_model_env.py)。位置：_wm_step; _compute_term_prob; _synth_sensor; sample_initial_states。
- **O-PROJECT** — [Official dependency specification](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/pyproject.toml)。位置：project.requires-python; tool.uv.sources。
- **G-WM** — [Go2 residual world model](https://github.com/Zhengtianjin/go2-supermimic/blob/d9eee9b3f248e5300e636c433255db2c1c3e086a/go2_supermimic/supermimic/world_model.py)。位置：WorldModel.forward; WorldModel.loss; RunningNorm。
- **G-IM** — [Go2 imagination](https://github.com/Zhengtianjin/go2-supermimic/blob/d9eee9b3f248e5300e636c433255db2c1c3e086a/go2_supermimic/supermimic/imagine.py)。位置：Imagination._step; _rewards; fall_prob; rollout; gae_targets。
- **G-TRAIN** — [Go2 alternating trainer](https://github.com/Zhengtianjin/go2-supermimic/blob/d9eee9b3f248e5300e636c433255db2c1c3e086a/go2_supermimic/supermimic/train.py)。位置：TrainCfg; improve_policy; run。
- **G-README** — [Go2 task and reproduction limitations](https://github.com/Zhengtianjin/go2-supermimic/blob/d9eee9b3f248e5300e636c433255db2c1c3e086a/README.md)。位置：Interfaces; How the released policies were trained; Limitations。
- **H-MIGRATION** — [Local runtime documentation](/home/qyw/MyHIMLoco/isaaclab_migration/README.md)。位置：当前运行环境; 可视化训练与导出。
- **H-ENTRY** — [Local training entry](/home/qyw/MyHIMLoco/isaaclab_migration/scripts/train.py)。位置：AppLauncher; stage/task parsing; Runner selection。
- **H-CONFIG** — [Local configuration loading](/home/qyw/MyHIMLoco/isaaclab_migration/himloco_lab/config.py)。位置：load_config; asset_contract。
- **H-CFG** — [WBLWBL walking configuration](/home/qyw/MyHIMLoco/isaaclab_migration/himloco_lab/legacy/wblwbl_dog_v1/wblwbl_dog_v1_config.py)。位置：WblwblDogV1Cfg; WblwblDogV1CfgPPO。
- **H-BASE** — [Inherited configuration](/home/qyw/MyHIMLoco/isaaclab_migration/himloco_lab/legacy/base/legged_robot_config.py)。位置：env; domain_rand; sim; normalization。
- **H-ENV-CFG** — [Isaac Lab environment configuration](/home/qyw/MyHIMLoco/isaaclab_migration/himloco_lab/env_cfg.py)。位置：HIMEnvCfg; make_env_cfg。
- **H-ENV** — [Lab lifecycle and actuation](/home/qyw/MyHIMLoco/isaaclab_migration/himloco_lab/env.py)。位置：sync_state; _pre_physics_step; _apply_action; _get_dones; _get_rewards; _reset_idx; _get_observations。
- **H-CORE** — [Core state and external events](/home/qyw/MyHIMLoco/isaaclab_migration/himloco_lab/core.py)。位置：HIMCore.__init__; _push_robots; _disturbance_robots。
- **H-LOGIC** — [Legacy observations and rewards](/home/qyw/MyHIMLoco/isaaclab_migration/himloco_lab/logic.py)。位置：check_termination; reset_idx; compute_observations; compute_termination_observations; _post_physics_step_callback; _compute_torques; reward functions。
- **H-ADAPTER** — [Legacy Runner adapter](/home/qyw/MyHIMLoco/isaaclab_migration/himloco_lab/adapter.py)。位置：LegacyRunnerAdapter.step/reset。
- **H-ACTOR** — [Local HIM actor](/home/qyw/MyHIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py)。位置：HIMActorCritic.__init__; update_distribution; act_inference。
- **H-EST** — [Local HIM estimator](/home/qyw/MyHIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py)。位置：HIMEstimator.forward/encode; HIMEstimator.update。
- **H-PPO** — [Local HIM PPO update](/home/qyw/MyHIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py)。位置：act; process_env_step; update。
- **H-RUN** — [Local HIM runner](/home/qyw/MyHIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py)。位置：HIMOnPolicyRunner.learn。
- **P1** — [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480v2)。位置：Section 3.1–3.3, Equations 1–3; Appendix A–D, Algorithm A.1, Tables A.4–A.6; Limitations。
