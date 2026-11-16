# AMP + HIE + SuperMimic：实现与实验方案

## 当前实现边界
当前可运行入口是 `him_supermimic/scripts/train_amp_hie.py`：Isaac Lab / PhysX、WBLWBL Dog V1、Stage 3 平地、随机初始化 HIM Actor/Critic/历史编码器。它先实现 AMP+HIE+PPO，不包含 SuperMimic 更新，不加载预训练权重。原 TrackerLab 入口保持独立。

复用 third_party/MyHIMLoco 的环境、动作/观测定义、HIMActorCritic、HIMPPO 和 HIE 辅助训练；复用 third_party/AMP 的 AMPDiscriminator（通过文件加载避免两个 rsl_rl 包冲突）。AMP 子模块来源为主目录 AMP 的已提交版本，远端为 MyAMP.git，未复制其未提交修改。迁移机器时先执行 `git submodule update --init --recursive`，另行安装 Isaac Lab 与 Python 依赖。

## 数据和奖励
读取 amp_supermimic/data/wblwbl/{walk,turn}/*/train.json（专家数据仍由 amp_supermimic 维护） 与 CSV；校验19列、有限值、关节名称，按仿真 dt 重采样，位置差分得到速度。不跨 clip 边界采样。walk/turn 各50%，组内 clip 均匀，再均匀选帧。参考特征均值和标准差固定并保存。

当前判别器特征为12关节位置+12速度，相邻两帧共48维；仿真下一帧必须取 reset 前状态，防止把重置跳变学成步态。此版本没有根部速度、足端位置和接触特征，不能把转向参考等同于“必然会转向”；转向方向和速度由任务奖励约束。

判别器采用 LSGAN（专家+1，策略-1）及专家梯度惩罚。策略先用固定判别器生成本轮奖励，再进行 PPO/HIE 更新，最后更新判别器。总奖励：

`r_total = r_task + amp_weight * dt * max(0, 1 - (D-1)^2/4)`

任务奖励沿用 controlled_flat 配置，foot_clearance、joint_power 关闭，其余有效项完整记录。实际生效权重见每次输出 config.json 的 legacy.rewards.scales（环境内部再乘dt）。AMP默认权重1，命令 x∈[-.5,.8]、y∈[-.2,.2]、yaw∈[-1,1]；平移范数≤.2清零。替换旧环境80%机器人硬编码x范围的采样器，确保配置与实际一致。关闭配置列出的域随机化，不代表初始姿态随机性完全关闭。

## 一键运行
```bash
cd /home/qyw/supermimic
bash him_supermimic/scripts/launch_amp_hie.sh train
# 另一终端看曲线
bash him_supermimic/scripts/launch_amp_hie.sh tensorboard
# 训练后，填写输出目录；默认比较随机初始和最后策略
bash him_supermimic/scripts/launch_amp_hie.sh play /绝对路径/训练目录
```
默认4096环境、3000轮、每轮24步，每100轮保存并评估。可追加 `--num-envs 1024 --iterations 500 --amp-weight 0.5 --seed 511`。HIM_PYTHON 可指定迁移电脑的解释器，AMP_OUTPUT 可指定输出目录（必须不存在）。目前不支持原地恢复训练，不要将重启当成续训。

TensorBoard 包含 reward_terms/* 全部有效加权逐步项、reward/task/amp/total、episode/* 完整回合累计值、PPO/HIE/判别器损失、判别器真假分数、动作std、跌倒、吞吐、物理交互数、评估速度误差与跌倒率。未结束回合不会产生 episode 曲线。零权重项没有有效奖励曲线。progress.jsonl 保存逐轮数据，checkpoint保存策略/编码器/判别器/归一化和优化器；不是完整仿真快照。

## 分阶段实验与验收
1. 管线检查：16环境3轮，验证所有损失有限、策略参数更新、reset前特征对齐、检查点可加载、TensorBoard有效项齐全。此阶段不能宣称会走。
2. 基线筛选：1024环境500轮；AMP权重0、0.5、1，种子511，其余配置固定。权重0为同入口HIM-PPO对照（仍有判别器开销，不能用于最终速度比较）。根据任务指标和视频选择AMP权重，不能按总奖励跨权重比较。
3. 基线确认：4096环境3000轮，种子511/512/513；分别评估前进(.5,0,0)、左转(.2,0,.6)、右转(.2,0,-.6)，每项至少20秒，独立种子801/802/803。当前训练内评估为种子801、10秒，仅作趋势监测；最终多种子长时评估需扩展。建议门槛：平面速度RMSE≤.15m/s、yaw RMSE≤.2rad/s、跌倒≤.1次/机器人分钟，三个方向全部通过，并用同命令视频检查拖脚/滑移/站立作弊。
4. 世界模型：从通过基线的真实交互采集状态、动作、历史、命令及reset前终态。按整条轨迹/种子拆分训练验证集，禁止跨reset训练。复用him_supermimic/dynamics.py等，报告1/8/16/32步状态、足端及任务奖励误差；动作库没有可靠扭矩/动作标签，不能直接当作真实动力学监督。
5. 可微AMP：冻结判别器参数和归一化，用普通前向保留状态梯度；不能调用带no_grad的predict_amp_reward。task+AMP奖励与真实环境逐项对照。HIE冻结参数但允许输入梯度，世界模型同理；验证Actor梯度、有限差分方向与真实短程回报。
6. SuperMimic消融：固定已通过的基线起点，比较继续PPO、仅BPTT、PPO+BPTT三组。先想象2/4/8步；不确定性门控、真实状态起点、终止mask、Actor更新约束逐项消融。判别器只在真实仿真/参考数据上训练，想象更新期间冻结，防止模型和判别器共同被策略利用。
7. 最终速度实验：种子511/512/513，从相同随机初始策略开始，统一奖励/动作库/指令/性能门槛；计入世界模型冷启动、判别器训练、真实采样和评估总时间，记录首次达到并连续三次保持门槛的时间及物理步数。分别报告纯BPTT与混合版，不把热启动结果当作从零加速。效果保持以各方向误差不超过基线10%、跌倒不增加为候选标准；无法达标就报告未证明提速。

## 后续代码边界
世界模型和BPTT复用 him_supermimic 的 dynamics、flat_task、him_policy、imagination；需新增共享AMP特征模块、可微AMP奖励适配和在线交替runner。此文档描述的后续阶段尚未实现；不要把本入口的PPO称为SuperMimic。HIE目前是本体历史表征，不自动等于复杂地形感知，复杂地形需独立验证观测与表征质量。
