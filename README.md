# WBLWBL SuperMimic × HIMLoco

目标：在已有 `/home/qyw/MyHIMLoco` 的 Isaac Lab 迁移环境中，为 **WBLWBL Dog V1 正常行走任务**引入 SuperMimic 的可微世界模型与短时域策略训练，保留 HIM 历史估计器和部署输入。

当前交付是**已初始化的集成仓库、两个已拉取并固定提交的源码子模块、源码审计和实施计划**。训练器尚未实现，未启动仿真或训练，未证明加速；当前仓库没有 GitHub 远端。现有 MyHIMLoco 仅作为本地只读依赖检查，未被修改，也未将其未提交代码复制进本仓库。

## 阅读入口

1. [原理与官方／Go2 代码导读](docs/01-principles-and-code.md)：模型学什么、BPTT 梯度、软存活、Critic、真实训练循环。
2. [WBLWBL 与现有 HIMLoco 接口审计](docs/02-himloco-audit.md)：270 维历史观测、关节映射、延迟、reset 和梯度截断。
3. [详细实施计划](docs/method-design.md)：推荐主路线、模块、阶段门槛、实验、失败处理与备选路线。
4. [逐阶段任务表](docs/training-plan.csv) 与 [计划参数](configs/wblwbl_dog_v1.plan.json)：参数均标明出处，计划配置目前没有训练入口消费。
5. [可点击源码索引](docs/source-index.md)、[固定版本与证据账本](docs/sources.json)、[本地源码快照元数据](manifests/himloco_source.json)、[上游版本锁](manifests/upstreams.json)。

## 已有目录

```text
third_party/supermimic_official/  # HybridRobotics/supermimic，Git 子模块
third_party/go2_supermimic/      # Zhengtianjin/go2-supermimic，Git 子模块
configs/                       # 计划参数，非可运行训练配置
docs/                          # 原理、审计、方案和实施任务
manifests/                     # 固定提交、实际配置和本地文件哈希
scripts/                       # 不启动仿真的仓库检查／源码指纹工具
```

未来实现使用 `src/him_supermimic/` 包名，避免与上游 `supermimic` 包冲突。不要直接把整个上游 Runner 安装进 HIMLoco 环境：官方要求 Python 3.13，并引用相邻 `mjlab_research`，本地迁移版文档记录 Python 3.11 / Isaac Lab 2.3.2 / PyTorch 2.7。两个子模块在本阶段用于源码对照，并不代表上游运行依赖已安装完整。

## 当前可以执行的命令

在本仓库根目录运行：

```bash
git submodule update --init --recursive
python3 scripts/check_workspace.py
```

第一个命令复现父仓库记录的 gitlink 提交；不要以 `--remote` 自动追随上游。第二个命令只检查源码与版本，成功不表示训练或仿真通过。

本机源码变化后，可主动更新审计快照：

```bash
python3 scripts/snapshot_himloco.py --source /home/qyw/MyHIMLoco --output manifests/himloco_source.json
```

它只导入不依赖仿真器的配置模块，读取 Git 状态、文件哈希及机器人资产契约，不运行 Isaac Sim，也不加载 checkpoint。重新生成后请审查差异；快照包含本机路径，分享仓库前可以替换成本人的路径。

## 实施顺序

推荐先在相同平地任务中建立 HIM-PPO 基线，完成数据接口和模型离线验证，然后接入 BPTT。先保留 HIM 估计器的真实数据辅助学习，再研究联合更新。完整地形、低趴、AMP 和 WMP 不进入首个 WBLWBL 行走里程碑。

## 许可证与代码来源

官方子模块保留 Apache-2.0；Go2 子模块保留其 MIT 以及机器人资源的独立许可。MyHIMLoco 根 README 标示 CC BY-NC-SA 4.0，部分代码文件另有 BSD/NVIDIA 声明。本仓库目前没有复制这些算法源码；后续移植需逐文件保留原有许可和署名，不将依赖统一改写成一个许可证。技术设计中的改造方案尚未经过实验验证。
