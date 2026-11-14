# A2 平地观测、指令和奖励对照

`him_supermimic/flat_task.py` 提供纯 Torch 的 WBLWBL 平地任务接口。接口读取 A1 的 37 维状态、指令及动作历史，可在连续轨迹上保留状态到观测和奖励的梯度。

## 时序和数据

1. 模型用本步裁剪后动作预测下一状态。
2. 指令在物理步后更新。重采样条件是 `(episode_step + 1) % period == 0`。外部采样负责提供已经应用旧环境死区的速度和目标航向。每步使用下一状态姿态重算偏航速度，应用原有环绕与限幅。
3. 奖励读取下一状态与更新后的指令。关节加速度使用控制周期两端的速度。平滑项使用本步与前两步的裁剪动作。每项保留原权重和控制周期缩放。
4. Actor 单帧按速度指令、角速度、投影重力、关节位置偏差、关节速度和动作拼接。新帧放到 6 帧历史的最前面。Critic 在单帧后附加机体线速度、外力和 187 维平地高度扫描。
5. 发生自动重置时，Critic 对照使用 A1 保存的终止观测。Actor 返回的新帧已经属于下一回合，验证仅比较其历史移位，Actor 单帧数值对照使用同回合数据。

`actor_frame` 与 `critic_frame` 支持外部传入噪声张量。单位与拼接顺序和 HIMLoco 一致。原回放缺少原始噪声采样，因此带噪数据按配置的噪声幅度检查，无噪数据做逐项数值检查。旧环境的终止 Critic 观测在常规观测裁剪前捕获，对照保留这一区别。

## 已运行的对照

| 数据 | 转移数 | 指令最大绝对差 | 撤去噪声后观测最大绝对差 | 奖励分项最大绝对差 | 总奖励最大绝对差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 受控平地，原带噪回放 | 384 | 1.19e-7 | 各块满足噪声范围 | 7.45e-9 | 1.12e-8 |
| A1 原配置，原带噪回放 | 384 | 0 | 各块满足噪声范围 | 3.73e-9 | 原配置含后续足端项 |
| 新采集受控平地，关闭噪声 | 288 | 0 | 0 | 4.77e-7 | 9.54e-7 |

无噪声记录包含一次接触终止、一次超时和一次指令重采样。动作与历史移位在三组回放中均精确对齐。状态和动作的自动微分均为有限非零值。双精度中心差分验证中，三组方向导数的最大绝对误差小于 `5e-12`。

`core_reward_v1` 的 11 项均从状态和动作重建。原配置的 `joint_power` 使用既有回放中最后物理子步的力矩，误差为 `2.91e-11`。后续完整奖励接入需要足端运动学，以及想象轨迹中的末子步力矩来源。周期性推力会导致 A1 保存的线速度与旧环境计算奖励时使用的缓存存在时序差别，验证器跳过该类记录并统计行数。本次三组排除行数均为 0。

## 复现

在仓库根目录运行。

```bash
python scripts/validate_a2.py outputs/a01/controlled/replay.pt
python scripts/validate_a2.py outputs/a01/untouched/replay.pt

python scripts/train_him_ppo.py --headless --device cuda:0 \
  --profile core_reward_v1 --num-envs 12 --iterations 2 --rollout-steps 24 \
  --verify-replay --no-observation-noise --output outputs/a2/reproduce
python scripts/validate_a2.py outputs/a2/reproduce/verification_replay.pt
```

本机报告保存在 `outputs/a2/a1-controlled.json`、`outputs/a2/a1-untouched.json` 和 `outputs/a2/noise-off.json`。后一份报告包含无噪声仿真回放的路径。
