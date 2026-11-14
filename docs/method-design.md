# WBLWBL Dog V1 接入 SuperMimic：详细实施计划

版本 1，2026-11-14。用户已确定机器人为 WBLWBL Dog V1；本计划推荐方案 A 作为主线。所有新模块和实验均为**待实现／待验证**，本次完成的是仓库、上游拉取、源码审计与计划。不存在已跑通的 `train_supermimic.py` 命令。

## 方向判断与边界

判断为“局部微调”：保留用户的 HIMLoco、WBLWBL 资产和 Isaac Lab 迁移环境，将新增工作集中在动力学模型、闭环想象和策略更新，而不是再次迁移物理后端。首个任务是正常高度行走，固定 stage=3 平地；低趴、粗糙地形、AMP 和 WMP 留到后续里程碑。

|原先泛化设想|本次源码核查后的具体方案|依据|
|---|---|---|
|给 PPO 添加世界模型|新建独立训练器，物理采样与想象 BPTT 分离|O-POL, H-RUN|
|直接调用 HIM 推理接口|为 estimator.encoder 增加保留输入梯度的包装接口|H-ACTOR, H-EST|
|读取 env.step 返回值构建 replay|在自动 reset 前截取完整状态，另保留 reset 后观测|H-ENV, H-ADAPTER|
|复用 Go2 环境参数|复用思路，按 WBLWBL 关节、45×6 观测、随机延迟重新适配|G-IM, H-CFG, H-LOGIC|
|直接安装官方包|源码子模块固定版本，新模块兼容现有 Python 3.11|O-PROJECT, H-MIGRATION|

`reported` 表示上游已实现机制；`adapted` 表示迁移到本任务；`proposed` 表示本计划提出、尚无直接实验支持。出处 ID 对应[源码索引](source-index.md)和 `sources.json`，阶段 ID 同时用于 CSV 与 handoff。

## 三种实质不同的接入方式

|方案|机制|部署|工程成本与风险|建议|
|---|---|---|---|---|
|A：HIM Actor 直接 BPTT|Actor 在模型中更新；HIM estimator 仍由真实仿真辅助目标更新|保留 HIM encoder+actor，270→12|需观测／奖励闭环，存在模型利用与估计器分布漂移|推荐首先实现；状态 provisional|
|B：冻结 HIM 基础策略＋残差 BPTT|基础策略冻结，仅优化有界残差动作网络|基础策略+残差，输入仍270维|可限制策略偏离，但需验证残差幅度、额外推理成本；不能保证单调改进|已有可靠 WBLWBL checkpoint 时备选；provisional|
|C：特权 BPTT 教师＋HIM 学生蒸馏|教师在模型中学习；学生在物理仿真状态上拟合教师动作，必要时真实 PPO 修正|只保留 HIM 学生|两阶段成本，特权信息不可辨识造成学生性能上限|直接历史梯度不稳定时研究备选；provisional|

三者均是本地组合方案，官方论文和 Go2 代码没有验证这三个 WBLWBL 配置。A 推荐不等于用户已经选择全部算法细节；handoff 的 selected_scheme_id 保留 null，recommended_scheme_id=A。

## 主线 A：模块与数据契约

以下是**计划目录**，尚未创建训练实现：

```text
src/him_supermimic/
  adapters/himlab.py       # Lab subclass、reset 前快照、HIM 导入边界
  contracts.py            # StateBatch / Transition / RobotSpec / config hash
  observations.py         # 45维单帧、270维最新帧优先历史
  commands.py             # heading→yaw rate、重采样事件
  models/dynamics.py      # 状态增量模型、旋转组合、上下文
  models/him_actor.py     # 保留 HIM 权重键的可微 forward
  models/value.py         # 想象状态的特权 Critic
  geometry.py             # WBLWBL FK/Jacobian，单位与关节映射
  rewards.py              # core_reward_v1 与 full_reward_v1
  termination.py          # 接触终止代理、概率校准
  replay.py               # episode-aware 窗口、真实历史 seed
  imagination.py          # 动作→状态→新历史→奖励，H步闭环
  learners/world_model.py # 真实日志多步监督
  learners/actor.py       # BPTT、冻结范围、梯度裁剪
  learners/value.py       # TD(lambda) 固定目标回归
  runner.py               # 采集/模型/估计器/Actor/Critic调度
  checkpoints.py         # 版本、优化器、统计、RNG、配置和数据状态
scripts/
  train_supermimic.py     # 将来：AppLauncher启动后才导入环境
  evaluate_policy.py      # 将来：只在物理环境评估
  export_policy.py        # 将来：encoder+actor，270→12
```

包名避免与上游 supermimic/rsl_rl 冲突。通过显式路径配置导入现有 `himloco_lab` 和本地 `rsl_rl`，不安装另一个官方 RSL-RL 覆盖用户版本。不直接 import Go2 的任务、机器人常量或 Actor。实际启动前还要处理本地 `make_env_cfg/prepare_urdf` 默认写入原工程缓存的行为：在新入口构建等价资产配置，将转换缓存、USD、日志和checkpoint显式放到本仓库 `.cache/outputs`，不依赖原工程目录可写；数值配置保持一致。

### 状态与 replay（A1，adapted）

先采用具名结构，不把所有信息藏进没有 schema 的张量：

|字段|建议形状|用途|
|---|---|---|
|root_pos_local / quat_xyzw|B×3 / B×4|相对环境原点的平移、姿态|
|base_v_body / base_w_body|B×3 / B×3|无观测缩放的真实物理速度|
|q / qd|B×12 / B×12|策略关节顺序，单位 rad、rad/s|
|actor_history|B×270|采样时实际历史，最新帧优先，含当时噪声|
|raw_action / executed_action|B×12 / B×12|带探索的输出与裁剪后命令|
|prev_action / prev_prev_action|各 B×12|执行器延迟与奖励历史|
|command / command_phase|B×4 / B|含目标 heading 与重采样时钟|
|context|B×C，按 RobotSpec 定义|摩擦、Kp/Kd、有效质量／COM、恢复系数等|
|event_context|具名字段|delay_substep、push、外力及作用时刻|
|next_state_pre_reset|同 state|真实下一状态，不用 reset 后状态替代|
|terminated / truncated / episode_id|各 B|终止、时间截断、序列边界|
|reward_terms / terminal_contact|具名字段|数值对齐和终止头监督|
|next_critic_obs_pre_reset|B×238|HIM 原辅助训练目标|

每个训练窗口包含 K 条动作和 K+1 个状态，按 episode_id 检查，不跨 reset。时间截断可以使用 reset 前末状态 bootstrap，死亡不能使用 reset 后新状态 bootstrap。物理模型不拟合“死亡→重置站姿”的跳变。PPO 原有 timeout 处理保持作为基线，新的目标处理需明确另行实现。[H-ENV `_get_rewards`; H-RUN `learn`; O-RUN `_step_and_store`; G-WM `loss`]

工作树包含未提交修改；每次实验记录 source HEAD、关键文件／资产 SHA256、两个子模块 SHA、配置和噪声种子。当前 manifest 是元数据，不是完整源码归档，跨机器复现前仍需用户版本化本地修改与资源。

### 动力学与奖励（A2/A3，adapted/proposed）

物理 state 合计37维（3+4+3+3+12+12）。第一版预测位置增量、旋转向量增量、速度与关节增量；旋转向量与四元数组合，避免直接累加四元数导致非单位旋转。translation 不应学习绝对环境编号；输入使用相对原点的位置，后续再对照 heading-invariant representation。

模型输入还包含历史动作和有效上下文。使用真实仿真轨迹更新归一化统计，在每次想象更新中固定统计。`L_WM=sum valid*weighted_state_error`，另可增加末子步 torque 监督和接触终止 BCE；这些额外头属于本项目设计，不是宣称 Go2 已有。[O-WM `_compute_losses`; G-WM `forward`/`loss`]

先实现 core_reward_v1：速度跟踪、垂向速度、横滚俯仰角速度、姿态、关节加速度、base height、动作一阶/二阶平滑、dof_pos、rear_thigh_pos，保持原项权重和 dt 缩放。暂不计 foot_clearance 和 joint_power，两组使用同一删减目标。完整任务另有 untouched HIM-PPO 参考组，不能用不同目标的 return 直接比较。

full_reward_v1 恢复 FK 足端项和最后物理子步功率语义。足速需按原式变换：原函数扣除了基座线速度后旋转到 body frame，没有再减去角速度叉乘项；若改用另一种足速定义，应同时修改对照组而不能声称数值等价。功率采用有监督 last-substep torque head 或更细子步展开，并报告近似误差。[H-LOGIC reward functions]

终止建议学习真实 `base_link` 接触死亡概率，辅以显式几何诊断；不能照抄 Go2 的70°倾角阈值。终止头需要校准且可能被 Actor 利用，因此所有存活改善必须在物理仿真复测。初始受控平地中也可用平滑底盘间隙代理做消融，但不能称为原终止规则。[H-LOGIC `check_termination`; O-ENV; proposed gap: contact proxy validity]

### HIM 梯度与独立优化器（A4，adapted）

```text
imag_actor(history):
    parts = estimator.encoder(history)       # 不 detach history
    vel, latent = split(parts, 3, 16)
    latent = normalize(latent)
    return actor(concat(history[:, :45], vel, latent))
```

这是设计伪代码。新接口应与旧 act_inference 的数值输出一致，但输入梯度不同。保持 estimator 参数名和 Actor 参数名，能加载原 checkpoint 的这两部分；不要加载旧 PPO optimizer 到新 Actor optimizer。高斯 std 用于真实探索的独立配置，不让 PPO entropy 项误入确定性 BPTT。

|更新步骤|允许更新|冻结权重但允许输入梯度|明确停止梯度|
|---|---|---|---|
|模型拟合|dynamics、可选辅助头|不需要 Actor/Critic|真实 target|
|HIM 辅助学习|encoder、target encoder、prototypes|无|真实标签、Sinkhorn 目标按原实现|
|Actor BPTT|Actor MLP|dynamics、estimator、Critic|回放 seed；Critic 回归数据副本|
|Critic拟合|新 Critic|无|TD(lambda) targets、想象样本|

原 HIM `estimator.update` 需要 next_critic_obs 的固定布局。保留真实 reset 前标签，并记录估计器独立学习率，初期不继承 PPO 自适应 KL 调度到 BPTT。每次完成 backward 后再进行下一模块更新，避免在同一计算图内原地改变权重。[H-EST `update`; O-POL]

新 Critic 输入建议 `[state37, history270, command4, contextC]`。HIM 策略有记忆，忽略历史可能使同一物理状态对应不同未来动作；初版保留历史，之后再消融是否压缩为 estimator latent。其结构不强制复用原238维 Critic，所以旧 Critic checkpoint 默认不兼容，需单独预热。

### 想象与更新伪代码（A5，adapted）

```text
collect_physics(): 记录真实 transition；auto-reset前保存末状态
update_world_model(): K步日志动作、真实上下文、多步误差
update_him_estimator(): 真实 history 和 next_critic_obs_pre_reset
freeze(world_model, estimator, value) 的参数
seed = replay.sample_state_and_actual_history()
for t in range(H):
    a_raw = imag_actor(history)
    a = clamp_like_physics(a_raw)     # 仍是归一化动作；不是已缩放的关节目标
    s_next = world_model(s, a, context, previous_action, event)
    reward = reward_fn(s, s_next, a, action_history, command)
    done_prob = termination_proxy(s_next)
    command_next = update_command(s_next, command, clock)
    history_next = prepend(observe(s_next, a, command_next, noise), history)
    累加 gamma^t * survival * reward；更新 survival
    s, history = s_next, history_next
J = discounted_reward + gamma^H * survival * value(final_state)
step_actor(-mean(J))
固定本次轨迹的 TD(lambda) targets，单独回归 Critic
重新开放相应模块权重，继续采样
```

模型输入约定为裁剪后的归一化动作，训练标签来自同一动作经过原 scale/PD 后的物理转移；模型内不要再次把它当作已缩放位置。关节目标只在执行器或需要解析PD的奖励路径中按 `q0+0.25*a` 计算。真实前向动作裁剪当前是配置值100，不可擅自改成常见的[-1,1]后仍称原HIM基线。实际代码要按本地 callback 时间顺序计算命令与奖励；上述图只表示依赖关系，A2 数值对齐决定最终调用顺序。不能在模型预测后偷偷读真实环境的 feet_pos、contact_forces 或未来 obs 计算奖励。[O-ENV `_wm_step`; H-ENV lifecycle]

## 分阶段实施与验收

以下时间为开发工作量估计，不含训练排队与硬件故障；GPU 小时未知，需 A0/A3 测量。参数阈值均为本项目建议，先在开发集校准并锁定，不能看测试结果后调整。

|步骤|操作和产物|进入下一阶段的门槛|估计开发量|
|---|---|---|---|
|A0 基线|独立入口读取 WBLWBL config，建立 untouched 与 core_reward_v1 两个 PPO 基线；保存配置、评估、耗时|确认关节、控制周期、270→12、GPU可用；已有checkpoint必须先物理评估|0.5–1天|
|A1 数据接口|HIMLabEnv subclass reset前快照、episode replay、上下文捕获、源码指纹|死亡/超时/部分reset/命令切换不混样；无reset状态跳变标签|1–2天|
|A2 观测与奖励|纯Torch观测、heading、历史、core_reward_v1、FK准备|关闭噪声下观测对齐 max_abs≤1e-5；解析奖励 abs≤1e-4或rel≤1e-3；窗口顺序正确|1–2天|
|A3 模型离线|仅训练 WM；按episode/随机化条件划分train/val/test；报告1/8/16/32步误差|运动验证段8步误差优于持久状态预测≥20%；建议关节RMSE≤0.05rad、机身速度≤0.2m/s、倾角≤0.05rad；失败单列|2–3天|
|A4 梯度与Critic|可微HIM wrapper、冻结范围、短时域value回归|前向与旧接口≤1e-6；独立检查历史→encoder→action非零梯度；模型/估计器权重不随Actor step变化|1–2天|
|A5 闭环训练|采样→WM→HIM→Actor→Critic；先H=8再16/32|无非有限loss；想象收益提高时物理评估同步改善；连续2次评估跌倒率显著恶化则暂停Actor更新诊断|2–3天|
|A6 完整奖励与随机化|恢复足端/功率项、delay、外力和参数随机化，再地形|分项奖励对齐；随机事件可解释；与同配置PPO比较，禁止只对BPTT简化任务|2–4天|
|A7 评估与导出|3 seeds、固定测试命令/地形、TorchScript、断点恢复|达到同一质量门槛的walltime有收益；导出数值≤1e-5；恢复训练配置/RNG/统计一致|1–2天|

A3 的20%要求针对运动数据各主要通道，不能通过大量站立样本拉低平均误差；不能用这一条件替代 A5 闭环评估。终止模型另报告PR、校准和失败召回；若高风险状态预测错误，Actor 不应开始长时域更新。

独立梯度核对：在同一可微模型内做方向有限差分验证 autograd；另在可重置物理状态与固定随机数条件下比较小动作扰动的效果趋势。接触不光滑时不要求物理有限差分逐点吻合，分接触／非接触样本报告。前者查实现，后者查代理是否具有控制价值。

## 初始参数与显存预算

|参数|上游依据|本项目起点（未验证）|
|---|---|---|
|物理/控制dt|本地0.005/0.02 s|保持不变|
|并行环境|Go2默认4096|512，初次冒烟64；按实测显存扩大|
|Replay长度|Go2默认1000/环境|256/环境；保存真实历史seed，后续再压缩|
|模型MLP|Go2 1024×3；官方2048×3|512×3起步，1024×3消融|
|模型窗口|论文8步|显式8个transition / 9个state|
|模型batch / updates|Go2 2048 / 2|512 / 2|
|想象batch / H|Go2 8192 / 32|512 / 8→16→32|
|模型/Actor/Critic lr|上游常用2e-4|WM 2e-4；Actor 5e-5；Critic 2e-4|
|gamma / lambda|ActorCritic 0.99 / 0.95|沿用0.99 / 0.95|
|梯度裁剪|官方norm1、element0.1|Actor先norm1再element0.1|
|Critic拟合|官方5 epochs×4 minibatches|先沿用，成本大时独立消融|
|HIM估计器|原辅助损失|按本地配置独立更新；冻结于Actor更新内|
|归一化/compile|Go2 running norm、compile|先FP32、不compile，正确后单独测compile收益|

总显存 = 仿真常驻 + Replay + 网络/优化器 + BPTT activations + 峰值临时张量。Replay 粗估 `num_envs * buffer_len * bytes_per_transition`；BPTT激活随 `imag_batch*H*各层宽度` 增长。不可只看模型参数大小。记录 collection、WM、HIM、actor、critic、eval 各自耗时，首次编译时间另列；OOM先降想象batch，再降环境数，不能通过悄悄减少评估难度获得加速。

本次工具环境 GPU 查询失败，原项目文档的16GB GPU是历史记录而非当前可用性确认。A0需在实际运行终端复核。不给出未经计时的“几小时必然收敛”。

## 对照实验与成功判据

|实验ID|内容|回答的问题|
|---|---|---|
|E0|原样 HIM-PPO，完整奖励与原随机化|已有任务的真实参考水平|
|E1|HIM-PPO，core_reward_v1、受控平地|与A5完全相同任务的公平基线|
|E2|A方案，同E1任务，从零／共享checkpoint两组分别报告|策略更新方式是否改善time-to-quality|
|E3|A方案，保留/截断估计器输入梯度|历史梯度贡献|
|E4|H=8/16/32，其他预算固定|偏差/时域/计算成本权衡|
|E5|末端Critic开/关|恢复和平衡的远期价值|
|E6|A方案恢复full_reward_v1与全部随机化，对比E0|收益能否延伸到原任务|
|E7|B或C，仅在触发条件满足时|残差或蒸馏是否改善稳定性|

正式比较至少3个种子。报告每种子的速度RMSE、yaw误差、跌倒次数/机器人分钟、成功时长、足滑移、torque/action变化、最终回报以及置信区间。测试命令、噪声、扰动和地形预先固定；训练与评估分开，eval关闭探索但保留规定的传感噪声。

建议质量门槛：速度RMSE≤匹配PPO基线的1.05倍，跌倒率≤基线+0.1次/分钟；预注册后才看测试数据。加速结论只在达到相同门槛时比较walltime/GPU-hours，至少报告是否超过1.2倍以及方差；小于计时噪声的差异不宣称加速。失败训练记为未达到门槛，不从平均值中删除。预训练数据/策略成本单列，从零总成本也单列。

## 失败诊断与恢复

|现象|优先检查|处理|
|---|---|---|
|WM一步就差|关节顺序、quat、动作执行值、reset前状态、dt|暂停Actor，修数据；不靠加大网络掩盖|
|一步好多步漂移|窗口边界、状态缺少delay/外力、旋转累积|修上下文，短H，增加失败/接触样本|
|想象奖励升、物理跌倒增|模型利用、终止头漏报、奖励代理偏差|回滚Actor，保留可用新数据重新拟合WM，降低Actor lr|
|HIM梯度为零|forward/encode内部detach，外层inference_mode，错误history push|用专用前向路径和梯度探针|
|恢复后动作不同|normalizer、joint map、config、encoder权重、历史reset|加载完整契约；拒绝静默形状兼容|
|无法在16GB运行|并行仿真和BPTT峰值、重复history/critic存储|测量后降低batch，不能依据Go2默认值推断本机容量|

检查点包含 Actor、estimator encoder/target/proto、WM、Critic、独立优化器、normalizer、RNG、配置、源码hash、课程状态和replay schema。分两种恢复：exact resume需要buffer和环境状态；只有模型权重则是warm resume，重新填buffer/预热，不声称逐步复现。导出仅 encoder+Actor，输入270输出12；保持部署端历史管理和关节映射，世界模型与Critic不部署。

## 备选 B：有界残差策略（proposed）

B0 要先有经实际评估可靠的 WBLWBL HIM checkpoint。冻结基础策略 `a0=pi_HIM(h)`；残差网络初始化输出0，执行 `a=clip(a0+alpha*tanh(delta_theta(h)))`。建议alpha从0.05开始，单位是归一化action，映射到本机位置目标再乘0.25；这是实验起点，不是已验证安全限幅。

B1 复用A1–A3数据/WM/奖励；B2只对残差参数BPTT，基础HIM参数冻结但其输入导数仍保留。目标 `-J + beta*||delta||^2`，beta先选1e-3并做0/1e-3/1e-2消融，避免把约束收益与算法收益混淆。Critic另训，基础Actor optimizer不恢复。B3相同E1任务比较残差BPTT、残差PPO、原HIM；部署同时运行两个网络，计入延迟。alpha饱和或模型预测与物理收益背离时停止扩大残差。模型梯度来自O-POL，残差架构本身为新提案，不能引用SuperMimic作为已验证残差结论。

## 备选 C：BPTT教师与HIM蒸馏（proposed）

C0 用A1–A3数据训练物理模型；C1教师 `pi_T(s,c,xi)` 用BPTT训练，initial lr5e-5、H8起步，其部署不可用的输入只留在训练端。C2让HIM学生在物理仿真执行，教师在学生访问状态上标注动作，拟合 `L_BC=||pi_HIM(h)-aT||²`；HIM速度／对比辅助损失仍从真实转移获得。学生可从旧checkpoint初始化，但应记录成本；BC阶段先只更新Actor，然后将encoder辅助更新作为独立步骤。C3报告教师/学生差距，再选择是否在真实仿真用原HIMPPO微调。禁止只在教师自身轨迹离线拟合后宣称学生稳定。

教师、学生、可选PPO微调的总GPU小时全部计入；保留teacher、student和aggregate数据版本。学生训练部署仍270→12。若history无法辨认教师使用的特权条件，蒸馏可能存在不可消除误差；以学生物理表现而非教师imagined return选择checkpoint。这条路线尚未在本仓库验证，不自动进入主线A。

## 步骤出处与迁移差异

|step_id|性质|来源与位置|支持的操作|本项目改动／待验证项|
|---|---|---|---|---|
|A0|adapted|H-ENTRY main；H-CFG；H-RUN.learn|现有WBLWBL HIM-PPO基线|另加匹配的core_reward对照|
|A1|adapted|O-RUN._step_and_store；H-ENV._get_rewards/_reset_idx；G-WM.loss|真实数据采集、窗口拟合|新增pre-reset完整快照和事件字段|
|A2|adapted|H-LOGIC.compute_observations/reward functions；O-ENV._synth_sensor|状态重建观测与解析奖励|WBLWBL FK与原时间语义数值对齐|
|A3|adapted|P1 §3.2 Eq.1；O-WM._rollout；G-WM.loss|多步WM监督|37维WBLWBL state、辅助头与小batch|
|A4|adapted|H-EST.forward/update；H-ACTOR.act_inference；O-POL.update|独立表征学习、参数冻结|新增不detach历史的前向接口|
|A5|adapted|P1 Eq.2–3/Algorithm A.1；O-POL；G-IM.rollout|BPTT＋TD(lambda)交替|从motion/chase改为WBLWBL velocity|
|A6|proposed|H-ENV延迟/外力、H-LOGIC功率项仅提供任务事实|恢复完整任务|随机事件与辅助头组合有效性未验证|
|A7|proposed|H-ENTRY/export入口仅提供现有格式|公平评估与导出|质量门槛、成本规则、模型版本检查|
|B0–B3|proposed|O-POL为BPTT组件依据；H-ACTOR为基础策略接口|残差方案|冻结基础策略＋有界残差组合待验证|
|C0–C3|proposed|O-POL为教师优化组件依据；H-EST为学生辅助目标|教师学生方案|特权BPTT＋聚合蒸馏组合待验证|

完整来源、版本和阅读记录在 sources.json，逐阶段机器可读任务在 training-plan.csv，续作数据在 method-handoff.json。源码事实由本次阅读核查；迁移收益尚无实验支持。下一次实施应先做A0/A1并完成首个transition契约，不能跳过这些检查直接运行大规模训练。
