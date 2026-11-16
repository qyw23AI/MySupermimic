# TrackerLab SuperMimic 命令训练迁移到 Isaac Lab

**状态：已实现首版代码，CPU 检查通过，PhysX 实机短测待进行。** 实施范围为 WBLWBL Dog V1 的平地 walk/turn、WMP `stair_gap`、`stair_gap_ramp` 与 `gap_combos`。本次 GPU 正在使用，测试期间隐藏了 GPU。

## 运行入口

从仓库根目录执行。以下检查只读 URDF、专家文件及 WMP 源码，可直接运行。`PY` 指向同时具备 Isaac Lab 和 TrackerLab Python 依赖的解释器，本次 CPU 检查使用 `himloco` 环境。

```bash
PY=/home/qyw/miniconda3/envs/himloco/bin/python
CUDA_VISIBLE_DEVICES='' "$PY" -B -m amp_supermimic.isaaclab_migration --list-tasks
CUDA_VISIBLE_DEVICES='' "$PY" -B -m amp_supermimic.isaaclab_migration preflight
CUDA_VISIBLE_DEVICES='' "$PY" -B -m amp_supermimic.isaaclab_migration preflight \
  --task wblwbl_dog_v1_physx_wmp_gap_combos
CUDA_VISIBLE_DEVICES='' "$PY" -B -m unittest discover -s amp_supermimic/isaaclab_migration/tests -v
```

GPU 空闲后的首个平地短测使用下面的命令。它使用 16 个环境和 3 轮更新，目标是检查训练链路，平地效果需另行评估。

```bash
"$PY" -m amp_supermimic.isaaclab_migration train \
  --task wblwbl_dog_v1_physx_flat \
  --run-dir amp_supermimic/outputs/physx/flat-smoke \
  --num-envs 16 --iterations 3 --warmup-steps 16 --wm-only-iters 0 \
  --wm-window 4 --wm-batches 1 --wm-batch 16 --pi-batch 16 --horizon 4 \
  --amp-history 4 --hist 4 --amp-batch 16 --amp-updates 1

"$PY" -m amp_supermimic.isaaclab_migration evaluate \
  --checkpoint amp_supermimic/outputs/physx/flat-smoke/last.pt \
  --output amp_supermimic/outputs/physx/flat-smoke-eval --steps 1000
```

首版训练通过 `runtime.py` 在启动时把 TrackerLab 上游训练器的仿真创建、资产与专家入口接到本项目，使用上游的真实经验回放、世界模型、AMP、Critic 和想象式 Actor 优化。运行目录包含展开后的 `legacy_task_config.json`、PhysX 生效配置 `physx_config.json`、输入哈希和训练结果。默认采用本地四条训练参考及两条留出参考。WMP 高度观测来自课程三角网格对应的顶面查询，离散阶段奖励只在真实转移中计算。WMP 学习课程的完整续训、障碍离散事件的可微学习及横梁净空射线属于后续验证或实现项。

## 1. 现状与迁移目标

| 现有路径 | 本地代码 | 迁移中的角色 |
| --- | --- | --- |
| TrackerLab 动作跟踪 | `third_party/trackerlab_supermimic/scripts/rls/supermimic/train.py`，环境 `source/trackerLab/trackerLab/tracker_env/supermimic/env.py` 的 `TrackingSim` | 已有 Isaac Lab / PhysX。参考其仿真生命周期、关节映射和控制循环 |
| WBLWBL 的 TrackerLab 命令训练 | `amp_supermimic/scripts/train.py` → TrackerLab `scripts/rls/supermimic/train_terrain.py` → `CommandSim(TerrainSim)` | 当前使用 MuJoCo-Warp。迁移对象是这条链路，保留命令式 Actor、世界模型、Critic、经验回放、AMP 和想象更新的训练语义 |
| WBLWBL 的 Isaac Lab 资产参考 | `third_party/MyHIMLoco/legged_gym/resources/robots/wblwbl_dog_v1/` 的 URDF 与网格，`him_supermimic/himlab.py` 的 `RecordingHIMLabEnv` | PhysX 导入复用 URDF 与网格，已有环境用于比对关节映射和接触行为 |
| WMP 的 Isaac Lab 地形 | `/home/qyw/WMP/isaaclab_migration/wmp_lab/{terrain,competition,competition_tasks,env}.py` | 复用碰撞网格、课程参数、赛道监测与已有射线实现。WMP 的 A1 策略、训练器和世界模型另有数据契约 |

本地审查锚点为 TrackerLab `5d1af04`、MyHIMLoco `0fbb2c4`、WMP `b712a5b`。实施采用当前本地可复用的 Isaac Lab 2.3.2 / Isaac Sim 5.1 组合为起点，首次运行记录实际版本、Python、Torch、CUDA、仓库提交及必要资产和数据的哈希。WMP 路径由新入口显式配置，默认可指向 `/home/qyw/WMP/isaaclab_migration`。平地配置应能独立于 WMP 导入和运行。

## 2. 目录与入口

目录位于 `amp_supermimic/isaaclab_migration/`。

| 位置 | 职责 |
| --- | --- |
| `__main__.py`、`cli.py` | 提供 `preflight`、`train`、`evaluate` 子命令，保存生效配置、评估指标及检查点 |
| `config.py` | 加载 `amp_supermimic.legacy` 注册的本项目配置类，应用显式覆盖，校验并生成 PhysX、TrackerLab 与 WMP 所需的运行配置 |
| `runtime.py` | 把本项目资产和 PhysX 真实仿真接到 TrackerLab 命令训练器 |
| `backend.py` | 用 PhysX 实现命令 rollout 和重置，向训练端提供 TrackerLab 兼容状态、控制量及终止信息 |
| `robot.py`、`references.py` | 机器人关节/足端映射，专家轨迹校审与格式转换 |
| `wmp_terrain.py` | WMP 几何、坐标变换、地形查询、课程与任务事件适配 |
| `../legacy/wblwbl_dog_v1/` | 在现有机器人目录新增 PhysX 平地与 WMP 任务配置类，并在 `../legacy/__init__.py` 的 `TASKS` 注册 |
| `tests/` | 配置继承、覆盖与实际生效检查，以及数据契约、重置边界和 WMP 规则一致性测试 |

实现时先把 `train_terrain.py` 中仿真创建与命令训练的耦合点列出来，明确训练器依赖的字段和方法。优先通过窄接口接入 PhysX。若没有可注入的训练入口，只在新目录移植必要的编排代码，继续导入上游的网络、回放、MDP 和 AMP。训练和评估应共用后端与配置解析。WMP 接入先验证是否能直接复用 `CourseState` 的兼容接口；需要调整 A1 足底假设时，只移植必要逻辑，并记录上游出处和一致性测试。

### 在 `amp_supermimic/legacy` 内独立管理任务参数

借鉴 HIMLoco 的管理方式，包括 Python 嵌套配置类、子类继承并覆盖字段、任务名注册，以及每次加载实例化一对 `Cfg` / `CfgTrain`。配置模块、默认值、注册表和加载器均由 `amp_supermimic/legacy` 维护。迁移入口复用本项目的 `BaseConfig`、`TASKS` 和 `load_task_config()`。验收时，在没有安装 `himloco_lab`、没有 `him_supermimic` 历史运行输出的环境中，仍能列出任务、实例化配置并生成参数快照。MyHIMLoco 仅提供 URDF、网格及用于人工核对的参数参考。

| 继承层 | 本项目定义的配置 |
| --- | --- |
| 公共基类 | 现有 `SuperMimicRobotCfg` / `SuperMimicCfgTrain` 提供 SuperMimic 通用环境、算法和运行默认值 |
| 通用机器人与 PhysX 基类 | 在现有 WBLWBL 配置下定义本地 `WblwblDogV1PhysxCfg` / `WblwblDogV1PhysxCfgTrain`，显式设置 URDF、默认关节角、控制周期等共用字段，并清理继承而来的 Warp 专用资产设置和历史运行路径 |
| 任务子类 | `wblwbl_dog_v1_physx_flat` 和 `wblwbl_dog_v1_physx_wmp_stair_gap` 分别注册一对 `<任务>Cfg` / `<任务>CfgTrain`，只在各自子类覆盖地形、命令、奖励、专家池和训练预算等差异字段。新增坡道任务时继承已有 WMP 任务子类并覆盖赛道差异 |

| 配置组 | 参数归属 |
| --- | --- |
| `env`、`robot`、`control` | 环境数、设备、种子、回合上限、URDF/初始姿态、PD、动作缩放和物理步长/decimation，均在本项目配置类定义。PhysX 适配层读取配置并根据 URDF 校验关节映射、限位和碰撞几何 |
| `terrain`、`commands` | 地形类型、WMP 源码路径、赛道等级与升降档、指令范围与重采样。命令课程沿用 `commands.curriculum`，WMP 难度课程归 `terrain` |
| `domain_rand`、`rewards`、`amp` | 域随机化、任务奖励与 WMP 事件奖励、AMP 专家池、参考重置、奖励权重及判别器设置 |
| `algorithm`、`world_model`、`runner` | 学习率、折扣、历史与想象视野、采样和更新预算、训练轮数、检查点频率和输出目录 |

平地任务类显式指定 walk/turn 专家池、速度/转向指令与域随机化设置。WMP 任务类显式定义赛道选择、指令、事件奖励和课程阈值。初始参数对照本计划第 4 节记录的 WMP 规则，赛道网格及成功判定逻辑通过 WMP 适配层复用。

加载顺序为 **公共默认值 → 机器人配置 → 注册任务配置 → 显式 CLI 覆盖**。共用字段复用已有的 TrackerLab 参数映射，PhysX 与 WMP 字段由新入口映射到对应后端。共享注册表按后端过滤各入口的 `--list-tasks` 和 `--task` 候选，已有任务继续走当前 Warp 入口。适配后的控制步长、课程阈值、奖励权重和训练预算以实际生效值为准。配置校验应报告目标后端无法应用的设置，避免配置字段存在但训练行为未变化。`--list-tasks` 和配置检查可在启动 Isaac Sim 前执行。

沿用 `legacy_task_config.json` 保存任务名及继承展开后的 `env_cfg` / `train_cfg`。另写 `run_config.json` 保存 CLI 覆盖后且完成后端映射的实际值，包括归一化路径、派生时间步、实际生效奖励和课程模式。训练、评估及恢复共用同一配置解析。评估从检查点关联的 `run_config.json` 还原任务，显式接受评估种子、环境数和固定等级等评估覆盖，并验证模型输入和资产契约。

## 3. 机器人、时间和专家数据契约

| 项目 | 实施约定和验收 |
| --- | --- |
| 机器人与控制 | 从本项目 `robot` / `control` 配置读取 URDF 路径和控制参数，构建 PhysX 资产。策略关节顺序为 RR、RL、FR、FL，各腿 hip、thigh、calf。按名称建立资产与策略之间的索引，校验关节符号、限位、零位、动作裁剪、目标角、PD 及力矩/速度限制。本项目 PhysX 基类的首版参数参考 `amp_supermimic/configs/wblwbl_dog_v1.plan.json` 中的物理步 0.005 s、控制步 0.02 s、动作缩放 0.25 和参考 Kp/Kd 40/1，逐关节配置值在短测中核对。 |
| 37 维状态 | TrackerLab 状态为 `p(3), quat(4), v(3), w(3), q(12), qd(12)`，四元数 `wxyz`，根部线速度与角速度在世界系。HIM 回放是 `xyzw`、机体系线速度及角速度，位置减去了自身 `env_origins`。明确区分 PhysX 世界系、TrackerLab 任务系和 WMP 单元局部系，平移只应用一次。测试 90° yaw 下的速度方向及变换往返。 |
| 单步时序 | 对齐上游 `CommandSim.step` 的 `s/s1`、本步 `c/c1`、执行动作、上一动作及终止标志。核对命令先推进再计算奖励，以及定时重采样发生的位置。终止前 `s1` 给世界模型和 AMP；下一轮决策读取重置后的观测。记录 `terminated`、`truncated`、`done` 和回合编号，验证 bootstrap、历史窗口和回放窗口边界。 |
| 平地参考来源 | `amp_supermimic/data/wblwbl/{walk,turn}/*/train.json` 当前有 4 个 walk 条目和 5 个 turn 条目。turn 中包含合成圆/八字/方形及左右原地转候选。按清单读取 fps、关节名和四元数顺序，从 19 列姿态序列生成 TrackerLab 能消费的参考。分段重采样、保持四元数连续、计算速度并核对 yaw 符号。按来源轨迹分开训练和留出集，记录每类、各方向和合成来源占比。 |
| AMP 与世界模型输入 | TrackerLab 的 AMP 以 `amp_style.py` 和 `train_terrain.py` 的实际特征、历史窗口、归一化及奖励为准。`him_supermimic` 中 HIM+AMP 入口的 48 维两帧关节特征不属于本迁移，仅可作外部参照。用同一参考片段和 PhysX 回放对照特征单位、足端位置、前进和左右 yaw 方向。不跨专家片段或真实回合采样。专家姿态序列用于风格监督；世界模型监督来自实际执行控制动作的 PhysX 转移。 |
| 权重与恢复 | 资产哈希、状态布局、命令维数、历史长度、地形输入和 AMP 特征都应写入检查点元数据。首个 PhysX 基准从头开始。后续热启动单列实验，只有逐层和观测含义匹配才复用权重。实现恢复时记录优化器、归一化、随机种子、课程和回放恢复策略，并单独测试恢复后的首轮采样。 |

## 4. WMP 赛道复用

1. **物理几何。** 复用 WMP `make_course()`、`course_mesh()` 与 `Terrain` 的单元布置和网格拼接，在 PhysX 注册真实碰撞网格。首批 `stair_gap`、`stair_gap_ramp`，从固定 L0 开始。对照 WMP 的 L0、L6、L9 网格和出生朝向。L0/L6/L9 的组合净沟宽分别是 5/36/54 cm。首批通过后，再接 `gap_combos`、常规楼梯/坡、梅花桩、轮胎阵和带横梁场景。
2. **地形信息。** WMP `HeightQuery` 可提供顶面查询，覆盖平台、沟底、坡和轮胎孔；顶视查询与 PhysX 碰撞网格抽样对照。顶视投影只能保留每个 XY 位置的一层表面。横梁下方净空需要对同一三角网格做定向射线查询。现有 TrackerLab 命令 Actor 每帧输入 45 维本体信息和 187 点高度扫描（17×11），世界模型和 Critic 使用 54 点地形子采样（9×6）。初版按这个布局复现，记录偏移、相对高度定义和部署时的扫描来源。
3. **支撑与成功。** WMP 的 `CourseState` 在多处按 A1 足心减 2 cm 算足底。WBLWBL 先从碰撞球的足心偏移和半径求世界系足底，并用站立、边缘接触、侧碰和四腿单独接触检查 PhysX 接触力绑定与方向。保留 WMP 按足名区分前后腿的规则。普通成功用于课程与回合结束，严格成功另记。A/B 组合均按顺序给两级台阶各 +0.5，起跳台 0，正常落台 +0.5，B 首次有效下坡 +0.5，严格整段 +2。事件奖励单位按 WMP 原代码核对，课程事件计算只执行一次。严格接触违规仅取消严格资格，危险掉落和通用失败仍按各自规则终止。
4. **课程与日志。** `eval` 固定类型和等级，`learn` 从 L0 开始，按 WMP 现行普通成功窗口独立升降。先对照配置中的 50 次尝试、80% 升档、低于 40% 降档以及 20 s 回合上限。逐类型和等级记录尝试、普通/严格成功、失败原因、台阶/落台/下坡覆盖和难度前沿。组合赛道命令沿 +X，过终点后的停稳逻辑与平地随机 yaw 命令分别配置。
5. **想象更新。** 初期用真实接触和课程状态机给出指标和离散事件。可微策略更新沿用 TrackerLab 已有的连续命令奖励和 AMP。障碍阶段对地形查询、连续高度和净空特征做梯度及误差测试，按边界和课程类型报告。接触状态机和硬碰撞保留在真实仿真侧。如果障碍训练要让 WMP 阶段奖励参与 TrackerLab 的可微 Actor 更新，先实现并验证连续事件近似或学习型事件头，使用真实事件监督，检查终止、bootstrap 和策略梯度。只在真实侧记日志或累计事件奖励时，应标明 Actor 实际优化的奖励项。

## 5. 实施顺序与验收

| 阶段 | 执行设置 | 进入下一阶段的条件 |
| --- | --- | --- |
| P0 资产与接口 | 核对后端契约、资产、专家清单。CPU 测试任务注册、继承、重复加载隔离和 CLI 覆盖，再做 PhysX 单环境查看 | 仅凭本项目配置即可解析平地和 WMP 任务，无需 HIMLoco 配置包或历史运行 JSON。平地任务也可在 WMP 源码缺席时加载。实际控制与解析值一致，四腿映射、足端接触、姿态/速度转换和专家留出集通过 |
| P1 平地迁移短测 | 1–16 环境先测站立、定向指令、强制摔倒、强制超时及重置。再用 16–64 环境做短测，显式缩小 warmup、batch 和窗口，覆盖真实采样、世界模型、判别器、Critic、Actor 更新 | 各模块至少有实际有限更新，Actor 有限非零梯度。历史和回放通过边界测试，评估脚本能用生效配置重载检查点。核对运行快照中的环境数、轮数、域随机化、奖励和 AMP 权重，并记录与 Warp 的差异 |
| P2 平地训练 | 用 256–1024 环境先估算可用显存与吞吐。种子 511 先跑 500–1000 轮，评估后再确定 3 个种子确认实验。`him_supermimic` 的 HIM+AMP 训练规模不作为本迁移的预算参照。TrackerLab 另记录每轮真实步数、世界模型/AMP/Actor 更新量及总用时 | 前进 `(.5,0,0)`、左转 `(.2,0,.6)`、右转 `(.2,0,-.6)` 分别评估，每条 rollout 至少 20 s，建议评估种子 801/802/803，累计足够机器人分钟。初始候选指标为平面速度 RMSE ≤ 0.15 m/s、yaw RMSE ≤ 0.20 rad/s、跌倒 ≤ 0.1 次/机器人分钟。检查支撑足滑、原地站立和转向方向，分别报告任务奖励与 AMP 奖励 |
| P3 地形与课程短测 | 固定 L0，先单项 `stair_gap`，再 `stair_gap_ramp`。使用同一网格的预览、PhysX 接触、录制或脚本化轨迹验证。随后开启 `gap_combos` 学习课程 | 同一输入轨迹在 WMP 规则与适配后得到一致的普通/严格成功、阶段奖励和下一等级。覆盖前足违规、后足进入真实 gap、后足接触目标平台边缘、危险掉落、坡道成功及重置统计 |
| P4 障碍训练 | 从已验收的平地模型开始，在 L0 先做短训和显存测试。支持保持/降低平地风格权重的对照，观察其对跨越动作的影响。世界模型按整回合和地形等级留出，比较 1/8/16 步预测，再测试 2/4/8 步想象更新 | 复验固定难度和课程，按类型/等级比较普通成功、严格成功和失败分布。先报告 L0 结果，再用固定种子独立评估 L6。与 PPO 基线比较时统一真实交互预算、任务奖励、评估口径和资产，单列世界模型冷启动及训练总时间 |

当前长训参数仍待容量与短测结果。实施阶段将保存每次运行的配置、数据清单、逐项奖励、模型误差、耗时、回放指标和视频。沿用 WMP 长训的操作习惯，训练由系统终端显示实时日志，WMP 地形长训每 300 轮保存检查点。平地与 WMP 使用各自输出目录，评估显式指定检查点和配置。

## 6. 审查项

- 首版推荐保留 **TrackerLab 命令训练算法 + PhysX 后端**，HIM + AMP 训练在 `him_supermimic` 中独立进行，不作为本迁移的一部分。建议批准后先做 P0/P1，再提交短测结果。
- 首个 WMP 障碍推荐 `stair_gap` L0，之后扩展到 `stair_gap_ramp` 和混合课程。优先复用原几何与规则，机器人足底适配作为独立验收点。
- 本计划默认首版复现 TrackerLab 命令策略的地形输入。若后续部署限定纯本体观测或指定传感器，应单独确定 Actor 和 Critic 的输入，再比较训练结果。

5 项 CPU 回归测试已通过，覆盖配置隔离、专家池加载和 AMP 采样、重置前 37 维转移进入回放、WMP L0 阶段奖励和严格成功，以及课程升档。另用 CPU 模拟状态接入真实 TrackerLab 训练器，跑通世界模型、AMP、Actor、Critic 更新及检查点保存。GPU 容量、PhysX 接触和场景隔离、真实仿真训练更新及评估回放待实测。
