# WBLWBL SuperMimic × HIMLoco

本仓库为 **WBLWBL Dog V1 正常行走任务**接入 SuperMimic。训练入口默认从 `third_party/MyHIMLoco` 加载 HIM-PPO、Isaac Lab 环境代码与机器人资产。

方案 A 的 A0 和 A1 已通过小规模 GPU 仿真，A2 的可微平地任务接口已通过逐项回放对照。[A0/A1 复现记录](docs/a01-validation.md) · [A2 验证记录](docs/a02-validation.md)

## 阅读入口

1. [原理与官方／Go2 代码导读](docs/01-principles-and-code.md)：模型学什么、BPTT 梯度、软存活、Critic、真实训练循环。
2. [WBLWBL 与现有 HIMLoco 接口审计](docs/02-himloco-audit.md)：270 维历史观测、关节映射、延迟、reset 和梯度截断。
3. [详细实施计划](docs/method-design.md)：推荐主路线、模块、阶段门槛、实验、失败处理与备选路线。
4. [逐阶段任务表](docs/training-plan.csv) 与 [计划参数](configs/wblwbl_dog_v1.plan.json)：训练入口读取其中的 WBLWBL 接口约定和受控平地配置。
5. [可点击源码索引](docs/source-index.md)、[固定版本与证据账本](docs/sources.json)、[本地源码快照元数据](manifests/himloco_source.json)、[上游版本锁](manifests/upstreams.json)。

## 已有目录

```text
third_party/supermimic_official/  # HybridRobotics/supermimic，Git 子模块
third_party/go2_supermimic/      # Zhengtianjin/go2-supermimic，Git 子模块
third_party/MyHIMLoco/         # HIM-PPO、Isaac Lab 环境与机器人资产
configs/                       # 计划参数，非可运行训练配置
docs/                          # 原理、审计、方案和实施任务
manifests/                     # 固定提交、实际配置和本地文件哈希
him_supermimic/                # 仿真采集、回放缓冲区和数据检查
scripts/                       # HIM-PPO 入口、仓库检查和源码指纹工具
```

MyHIMLoco 子模块提供训练源码和资产。训练需要另行安装 Python 3.11、Isaac Lab 2.3.2 及匹配的 Isaac Sim、PyTorch。官方 SuperMimic 和 Go2 子模块用于源码参考。MyHIMLoco 内部的旧工程子模块无需初始化即可运行这里的平地训练。

## 初始化与迁移

克隆 SuperMimic 后，从仓库根目录运行以下命令，按父仓库锁定的版本初始化三个子模块。

```bash
git submodule update --init
python scripts/check_workspace.py
```

MyHIMLoco 的 WBLWBL 配置与网格均随提交 `88dafdf` 获取。启动训练时直接运行 `scripts/train_him_ppo.py`，默认使用这个子模块。

训练命令见 [A0/A1 复现记录](docs/a01-validation.md)。更新 MyHIMLoco 固定版本时，同步更新子模块引用与 `manifests/upstreams.json`，并刷新源码指纹。

```bash
python scripts/snapshot_himloco.py --output manifests/himloco_source.json
```

## 实施顺序

推荐先在相同平地任务中建立 HIM-PPO 基线，完成数据接口和模型离线验证，然后接入 BPTT。先保留 HIM 估计器的真实数据辅助学习，再研究联合更新。完整地形、低趴、AMP 和 WMP 不进入首个 WBLWBL 行走里程碑。

## 许可证与代码来源

官方子模块保留 Apache-2.0；Go2 子模块保留其 MIT 以及机器人资源的独立许可。MyHIMLoco 根 README 标示 CC BY-NC-SA 4.0，部分代码文件另有 BSD/NVIDIA 声明。本仓库目前没有复制这些算法源码；后续移植需逐文件保留原有许可和署名，不将依赖统一改写成一个许可证。世界模型和策略 BPTT 属于后续阶段。
