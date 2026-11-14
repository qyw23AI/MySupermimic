# WBLWBL Dog V1 平地基线与转移采集

在本机 Isaac Lab、PhysX 和 RTX 5060 Ti 上运行了两组方案 A 冒烟验证。每组 12 个环境、每轮 24 步、2 轮 PPO 更新，共 576 条训练转移。策略输入为 270 维，动作输出为 12 维，控制周期为 0.02 秒。每组额外评估 48 个控制步，并保留训练结束前 32 步的回放。这一规模用于验证训练链路和数据契约。

| 配置 | 训练转移 | 保存的转移 | 8 步有效窗口 | 状态连续性最大绝对误差 | 奖励分项求和最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原平地奖励及随机化 `untouched` | 576 | 384 | 182 | 0 | 1.19e-7 |
| 受控平地 `core_reward_v1` | 576 | 384 | 184 | 0 | 9.54e-7 |

有效窗口来自独立的 40 步验证，环形缓冲区保留最后 24 步。每组在一台机器人上触发一次超时，在另一台机器人上通过 PhysX 底盘接触触发一次终止，并触发一次命令重采样。校验比较了 reset 前后姿态、HIM 终止观测、同一回合相邻状态、实际裁剪动作及四个物理子步的动作顺序。采样窗口按环境和回合筛选，包含 8 个动作、9 个状态。验证脚本还逐个枚举合法窗口，与缓冲区的结果比较。

## 复现

在已安装 Isaac Lab 和 PyTorch 的环境中，从 SuperMimic 仓库根目录运行。MyHIMLoco 代码和资产从子模块加载。每次指定一个新的输出目录。

```bash
for profile in untouched core_reward_v1; do
  python -B scripts/train_him_ppo.py \
    --headless --device cuda:0 --profile "$profile" --num-envs 12 \
    --iterations 2 --rollout-steps 24 --replay-steps 32 --verify-replay \
    --output "outputs/a01/reproduce-$profile"
done
```

实际运行产物位于 `outputs/a01/untouched` 和 `outputs/a01/controlled`。目录包含 `model_2.pt`、`metrics.json`、`replay.pt`、`replay_validation.json`、生效配置和 HIM 源码指纹。URDF 转换文件与 USD 位于各次运行的 `asset_cache` 目录。

## 转移时序

- `state` 和 `actor_history` 在执行动作前采集。状态由相对环境原点的位置、xyzw 四元数、机体坐标系线速度和角速度、策略关节顺序的 q 与 qd 构成。
- `raw_action` 保存进入动作处理的值，`executed_action` 保存裁剪结果，`substep_actions` 保存实际送入四个子步的动作。延迟用实际序列表示。前后动作相同时序列仍准确。
- `next_state_pre_reset` 在本步奖励计算后、自动 reset 前保存。若该步触发 legacy push，状态包含已施加的速度更新，`push_applied` 标记此事件。`applied_force_body` 和 `pending_force_next_body` 分别记录本步施加的力和 reset 前下一步待施加的力。原配置的 push 周期未在本次短验证中触发。
- `actor_history_after_step` 保存 step 返回的观测。发生 reset 时它属于新回合。`next_critic_obs_pre_reset` 对已重置环境使用 PPO Runner 同样的终止观测替换规则。
- `episode_id` 与环境列共同标识回合。`command` 和 `command_pre_reset` 分别记录步前命令与奖励计算时的命令。`episode_step` 保存步前计数，`command_resampled` 标记本步是否重采样。默认窗口排除命令重采样，允许终止或超时落在窗口最后一步。
- 采集的物理参数包括摩擦、恢复系数、PD 因子、各刚体质量和质心。分项奖励已经乘原配置权重，按 `reward_term_names` 排列。配置保留名义增益、动作缩放和顺序。
