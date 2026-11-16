# HIM + SuperMimic

这里实现 WBLWBL Dog V1 的 HIM 学习和世界模型训练。默认任务是 MyHIMLoco 的 Stage 3 受控平地，物理仿真使用 Isaac Lab / PhysX。仿真子步 0.005 s，每 4 个子步执行一次策略控制。策略输入 270 维（6 帧），输出 12 个关节动作。

## 启动纯 SuperMimic 冷启动

先初始化 MyHIMLoco 子模块，并使用已安装 Isaac Lab 和匹配 Isaac Sim 的 Python 环境。下面是开发种子的一次训练，可以直接在系统终端启动。

```bash
cd /home/qyw/supermimic
/home/qyw/miniconda3/envs/himloco/bin/python -uB him_supermimic/scripts/train_cold_start.py \
  --headless --device cuda:0 --seed 511 --num-envs 512 --iterations 500 \
  --output him_supermimic/outputs/zero-start/b-seed511
```

输出目录须是新目录。若 Isaac Lab 无法打开默认地面 USD，可加 `--ground-usd /绝对路径/default_environment.usd`（同一份本地地面资产用于对照）。第一次只排查启动时，可以用 `--num-envs 64 --iterations 3 --eval-steps 16`，这会采样并训练模型，默认冷启动等待轮数意味着它不会进入 Actor 更新。

入口在本次运行中初始化 HIM Actor、编码器和世界模型。真实仿真用随机策略加探索噪声采样。模型用控制步末、reset 前的状态做 8 步多步拟合，并用保留的环境流检查线速度和偏航误差。两项 MSE 都低于状态保持基线的 90% 且过了最小预热周期，才尝试 BPTT。Actor 只沿想象回报做梯度更新。HIM 编码器另外使用原有的真实数据辅助损失更新。当前不使用 PPO、Critic 尾值或历史权重。

输出含 `config.json`、`initial_policy.pt`、`progress.jsonl`、`tensorboard/`、周期检查点和最后的 `report.json`。计时从启动到每次检查结束，覆盖仿真初始化、冷启动、模型优化和在线评估。当前入口是开发版。门控以留出预测误差为准，想象终止仍使用 A5 高度/倾角代理。物理评估使用 P0 指令套件的单个开发种子。正式 A/B 提速仍要校准真实梯度方向与接触终止，并接入 [实验方案](docs/wblwbl-zero-training-plan.md) 的完整达标判定、留出验证与统一预算。

## 奖励和任务参数

| 参数                                                 | 在哪里改                                                                                                                                    | 生效方式                                                                                                                  |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 基础奖励权重、跟踪奖励宽度、目标高度、各关节位置权重 | [MyHIMLoco WBLWBL 配置](../third_party/MyHIMLoco/isaaclab_migration/himloco_lab/legacy/wblwbl_dog_v1/wblwbl_dog_v1_config.py) 的 `rewards` | Isaac Lab 真实仿真与本仓库 `FlatTask` 都读取生效后的同一份配置                                                          |
| 受控平地覆盖项                                       | [wblwbl_dog_v1.plan.json](configs/wblwbl_dog_v1.plan.json) 的 `controlled_flat`                                                            | `omitted_terms` 中的奖励归零；`domain_rand_overrides` 控制扰动、延迟及物理随机化；`command_curriculum` 控制指令课程 |
| 指令范围、重采样频率、动作比例、PD、观测噪声         | MyHIMLoco WBLWBL 配置和基础配置                                                                                                             | 同时影响真实训练任务及可微重建；评估指令另由 `evaluation.py` 的 `schedules()` 生成                                    |
| 模型、Actor、HIM 更新预算与学习率                    | `train_cold_start.py --help`                                                                                                              | CLI 只改变学习与采样流程，任务参数写入每次运行的 `config.json`                                                          |

`core_reward_v1` 的 11 项为 `tracking_lin_vel`、`tracking_ang_vel`、`lin_vel_z`、`ang_vel_xy`、`orientation`、`dof_acc`、`base_height`、`action_rate`、`smoothness`、`dof_pos`、`rear_thigh_pos`。配置值是乘以控制周期之前的权重。例如线速度跟踪权重 `2.0` 在控制步上乘 `0.02`，得到 `0.04`。默认关闭 `foot_clearance` 和 `joint_power`。新增奖励项需要在 MyHIMLoco 的真实奖励和 [flat_task.py](flat_task.py) 的想象奖励中一起实现，然后用 [validate_a2.py](scripts/validate_a2.py) 对照其数值与时序。解除延迟、推力或其他随机化前，需要核对世界模型是否得到相应输入。做 A/B 实验时，两组使用完全相同的资产、指令和奖励覆盖。

| 冷启动参数                                           |             默认值 | 作用                                                                       |
| ---------------------------------------------------- | -----------------: | -------------------------------------------------------------------------- |
| `--num-envs` / `--iterations`                    |          512 / 500 | 并行环境、训练周期数                                                       |
| `--collect-steps` / `--replay-capacity`          |             8 / 32 | 每周期真实控制步、每环境保留步数                                           |
| `--model-updates` / `--batch-size`               |            8 / 256 | 每周期优化次数及窗口批量                                                   |
| `--model-lr` / `--actor-lr` / `--estimator-lr` | 2e-5 / 5e-6 / 1e-5 | 分别更新动力学、Actor、HIM 编码器                                          |
| `--exploration-std`                                |  继承 HIM 初始 std | 采样动作的额外高斯噪声                                                     |
| `--cold-start-min` / `--cold-start-max`          |           20 / 200 | 最早尝试 Actor 的周期、无接受更新时的停止预算                              |
| `--anchor-weight` / `--max-action-p95`           |        0.01 / 0.02 | 想象轨迹的动作偏移惩罚与真实回放起点动作 P95 上限；超限回退 Actor 和优化器 |
| `--eval-every` / `--eval-steps`                  |          100 / 320 | 物理评估间隔和每种指令的步数                                               |

## 文件关系与 MyHIMLoco 复用

```text
MyHIMLoco 任务配置 / URDF / Isaac Lab 环境 / 动作与 reset 逻辑
  → himlab.py：配置资产、在 reset 前采集状态和奖励
  → replay.py：按 episode 管理真实轨迹窗口
  → dynamics.py：用真实状态和动作拟合 WorldModel
  → imagination.py：将同一 HIM 策略与 FlatTask、WorldModel 展开
  → train_cold_start.py：拟合模型、HIM 辅助学习、BPTT 更新和日志
  → evaluation.py：在真实 Isaac Lab 用固定指令测策略
```

- **直接复用 MyHIMLoco**：机器人 URDF 和网格、WBLWBL Stage 3 配置、`HIMLabEnv`、原奖励和观测、PD/限幅、接触与超时 reset、`LegacyRunnerAdapter`。B 组还复用 `HIMActorCritic` 的 Actor 与 `HIMEstimator` 的编码器及训练损失，构建随机初始化策略。
- **本仓库补充**：`him_policy.py` 绕过原推理的 detach，保留想象历史的输入梯度。 `flat_task.py` 在 Torch 重建观测、指令与奖励，`termination.py` 提供软存活模型，`value.py` 为旧 A5 提供价值模型，`evaluation.py` 统一统计 reset 前的回报、速度误差和跌倒。
- **启动入口**：`train_him_ppo.py` 复用原 PPO Runner，使用 `--profile core_reward_v1` 跑同配置基线。 `train_cold_start.py` 启动 B 组。 `train_online.py` 用于有权重的 A5 微调。 `train_world_model.py`、`warmup_critic.py`、`compare_online.py` 分别服务离线模型、旧 Critic 预热和仿真回放。
- **HIM-PPO + AMP**（自 `amp_supermimic` 迁入）：`scripts/train_amp_hie.py`，由 `bash him_supermimic/scripts/launch_amp_hie.sh train | play RUN | tensorboard` 启动，默认输出 `outputs/amp-hie-<时间>`；专家动作仍读 `amp_supermimic/data/wblwbl`。方案见 [docs/AMP_HIE_PLAN.md](docs/AMP_HIE_PLAN.md)。早期 smoke 输出在 `outputs/amp-hie-smoke-0{1,2,3}`（其 `config.json` 内记录的是迁移前的旧路径）。
- **诊断与历史记录**：`scripts/diagnostics/` 包含可独立运行的检查点评估、Actor 更新探测及接触模型诊断。固定参数的旧批次入口已移除。阶段结果与旧实验方案见 [实验日志](docs/experiment-log.md)，算法原理、接口审计和上游出处见 [架构与源码索引](docs/architecture.md)。
