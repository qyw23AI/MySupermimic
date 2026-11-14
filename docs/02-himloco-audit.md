# WBLWBL Dog V1 × 本地 HIMLoco 源码审计

范围：`/home/qyw/MyHIMLoco` 工作树，2026-11-14。用户指定首个机器人为 **WBLWBL Dog V1**，选择正常行走 `wblwbl_dog_v1`；低趴任务为后续对照。本地 HEAD 和 20 份关键源码 SHA256 记录在 `../manifests/himloco_source.json`。工作树包含未提交修改和未跟踪 WBLWBL 配置／资产，单独记录 HEAD 不能复现本次审计。

## 已确认的实际接口

|项目|本地读取结果|证据|
|---|---|---|
|后端|已有 Isaac Lab 迁移版，直接导入本地修改的 rsl_rl|H-ENTRY, H-ADAPTER|
|物理／控制间隔|0.005 s × decimation 4 = 0.02 s，策略 50 Hz|H-BASE, H-CFG, H-ENV-CFG|
|动作|12 个关节；P 控制，scale=0.25，Kp=40、Kd=1；随机因子需另算|H-CFG, H-LOGIC `_compute_torques`|
|策略输入|45×6=270，最新帧在最前|H-BASE, H-LOGIC `compute_observations`|
|单帧布局|command 3，angular velocity 3，gravity 3，q-q0 12，qd 12，上一步执行动作 12|H-LOGIC|
|Critic 输入|238=45+3 线速度+3 外力+187 高度采样|H-BASE, H-LOGIC|
|HIM 输出|速度 3＋归一化 latent 16|H-EST|
|Actor 输入维数|45+3+16=64；隐藏层 512/256/128，ELU|H-ACTOR, H-CFG|
|PPO rollout|100 控制步/环境；不能与 Go2 的迭代数直接比较|H-CFG|
|行走高度|奖励目标 0.33 m；初始 root z=0.40 m|H-CFG|
|硬终止|指定底盘接触力范数 >1；另有时间截断|H-LOGIC `check_termination`|
|初始目标环境|stage=3 为平地；stage=4 才使用任务地形|H-CONFIG|

配置数字来自读取／解析，而不是本次仿真实测。关节顺序必须使用 `asset_contract()` 返回的列表（见 manifest），不能使用 default_joint_angles 字典的排列推断。HIMLabEnv 已通过 `joint_ids` 映射 Lab 关节与旧策略顺序。本次资产解析结果为 **RR → RL → FR → FL**，每条腿按 hip/thigh/calf 排列。正式运行仍要检查导入后的 `joint_ids`，不可照搬 Go2 或常见 FL/FR/RL/RR 顺序。来源 ID 的完整链接见[源码索引](source-index.md)。

## 需要特别处理的八处接口

### A. 梯度不是只在 Runner 外层被关闭

`HIMOnPolicyRunner.learn` 在 `torch.inference_mode()` 中采样；这对物理采样合理，但想象更新必须在其外。`HIMActorCritic.update_distribution` 在 no_grad 下调用估计器；`HIMEstimator.forward` 同时 detach 输入与输出，`encode` 也 detach 输入。

计划增加 `HIMDifferentiableActor` 包装器，在新仓库直接调用现有 `estimator.encoder`，切分并归一化输出，再调用原 `actor`。保持权重键兼容，冻结估计器权重但保留历史输入梯度；不在原仓库全面删掉 detach。只用 `act_inference` 仍会调用内部 detach，不能解决问题。[H-ACTOR, H-EST, H-RUN]

### B. 自动 reset 后的状态不是上一条转移的 next_state

Lab env 会在一次 step 内执行终止判断、奖励、reset 和新观测。当前 `_get_rewards` 仅保存 `terminal_ids` 和 `terminal_obs`；238 维 terminal critic observation 不等价于完整物理状态。

在新仓库继承 `HIMLabEnv`，在 reset 前的 hook 保存所有环境的 post-physics state、reward、terminated、truncated、上下文；再将 reset 后 policy observation 交给下一次采样。不得在普通 `env.step` 返回后才读取 root_state，当作死亡转移的 next_state。[H-ENV `_get_rewards`, `_reset_idx`; H-ADAPTER `step`]

### C. 现有 reset 没有显式清空整个历史

`LegacyLogic.reset_idx` 清空 last_actions 等缓存，但没有清空 `obs_buf`；新帧再被 prepend 到原历史。为公平基线，第一版保持现有行为并保存实际 270 维历史快照。物理模型的转移窗口仍必须按 episode_id 截断。不能凭历史窗口“应该如何”擅自改变基线。

若后续统一改成零填充／首帧填充，必须同时修改对照组，并作为独立实验记录；不能只改变 SuperMimic 一侧。[H-LOGIC `reset_idx`, `compute_observations`]

### D. 有两层动作时间尺度

`_pre_physics_step` 先 clamp 动作，再在四个物理子步之间随机切换上一动作和当前动作。delay 每控制步抽样，不能把它当作固定每 episode 的参数。模型必须知道这次实际执行的延迟，或对延迟显式边缘化；首个受控实验先对两组同时关闭 delay。

Replay 保存 noisy raw action、clipped action、上两次执行动作、delay_substep、控制参数。二阶 smoothness 需要 `a[t-2]`，只有 `a[t-1]` 不够。官方/Go2 的原动作语义不能代替这里的实际执行语义。[H-ENV `_pre_physics_step`; H-LOGIC `_compute_torques`, `_reward_smoothness`]

### E. 四元数约定已有转换

Lab 是 wxyz，legacy core 是 xyzw。`sync_state` 与 `write_root` 已转换。新模型在内部约定一个顺序并在边界转换；旋转预测用小旋转增量组合并归一化，不直接无限累加原始四元数。单位、参考系和关节顺序都写入 checkpoint schema。[H-ENV `sync_state`, `write_root`]

### F. heading command 不是永远固定的 yaw rate

当前配置 `heading_command=True`，第四维是目标 heading；每步根据当前姿态重新计算 `commands[:,2]` 并裁剪。若想象中一直保持起始三维指令，会与物理环境不同。

应保存四维命令、命令更新时间和 episode step，在预测姿态上重算 yaw 指令。第一版短时域禁止跨命令重采样边界，后续显式复制命令事件；也可将两组共同改为固定 yaw-rate 指令，但必须另命名受控实验。[H-LOGIC `_post_physics_step_callback`; H-CFG `commands`]

### G. 外力、随机 push 与物理参数

继承配置包含外力、根速度 push、摩擦、恢复系数、Kp/Kd 等随机化。有些随机量在 Python 配置中存在，却不一定真正乘入执行路径，例如本次读取的 P 控制公式没有乘 `motor_strength_factors`，不能仅凭开关就将它称为有效执行器随机化。

`_get_dones` 内 callback 会触发命令／push／外力事件；应区分 physics 后状态、事件后的下一决策状态和下一步待施加外力。对瞬间写入仿真器的 push，缓存 core 速度还可能需要同步。M1 同时关闭随机事件；后续捕获事件、时间和实际参数，不让确定性模型把未记录的跳变当成动力学误差。[H-ENV; H-CORE `_push_robots`, `_disturbance_robots`; H-LOGIC]

### H. 奖励不能只按名字复制

|现有非零奖励|可微重建要求|
|---|---|
|tracking_lin_vel / tracking_ang_vel|使用原 body-frame 定义、sigma=0.25 和更新后的命令|
|lin_vel_z / ang_vel_xy / orientation|从预测状态读取，保持观测缩放与物理奖励量分开|
|dof_acc|前后关节速度差除控制 dt，再平方|
|action_rate / smoothness|保留一阶／二阶动作历史，使用裁剪后动作|
|base_height|平地为地面相对高度；地形阶段复现采样函数|
|dof_pos / rear_thigh_pos|原关节权重与索引；不假定四条腿完全对称|
|foot_clearance|由 WBLWBL URDF FK/Jacobian 重建足位置和速度；原函数使用 body-frame 相对足高与平面足速，并非简单世界坐标抬脚高度|
|joint_power|原代码使用最后物理子步的 torque 缓存与当前 qd；以末状态再算 PD 会有时序差异|

奖励权重还会在 `_prepare_reward_function` 中乘控制 dt。首个算法实验可暂用明确的 `core_reward_v1`（暂不含 foot_clearance / joint_power），但 PPO 和 BPTT 两侧必须同时使用相同目标，并单独报告与原任务差异。恢复完整奖励时，FK 必须在真实状态上数值对齐；最后子步 torque 可增加辅助预测头，不能声称用末状态 PD 就严格等价。[H-CFG, H-LOGIC]

## 环境与资产状态

原工程 README 记录 RTX 5060 Ti 16 GB、Python 3.11.17、PyTorch 2.7.0+cu128、Isaac Sim 5.1、Isaac Lab 2.3.2。本次仅解析配置和资产契约，没有重跑这些运行验证；当前工具会话的 nvidia-smi 无法连接驱动，可能与会话访问条件有关，不能据此诊断宿主机驱动损坏。训练前应在实际运行终端复核 GPU、解释器和空闲显存。

官方 SuperMimic `pyproject.toml` 要求 Python >=3.13,<3.14，`uv.sources` 使用相邻 mjlab_research 路径。当前只拉取用户指定的两个子模块，不自动安装这个额外仿真栈。独立实现 `him_supermimic` 使现有 Lab 环境与源码参考可共存。
