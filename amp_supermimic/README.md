# amp_supermimic

WBLWBL Dog V1（12 自由度四足，关节顺序 `RR, RL, FR, FL`，每腿 `hip, thigh, calf`）的 **AMP 专家动作数据 + 命令式 SuperMimic/AMP 训练** 工程。
本目录只放"粘合代码、数据和文档"，算法与仿真内核来自 `../third_party/` 下的子模块（不修改它们）。

> 说明：本文档中的路径、命令行参数均对照仓库文件或 `--help` 输出编写。

## 1. 训练范围

| 内容 | 入口 | 仿真/算法 | conda 环境 |
| --- | --- | --- | --- |
| TrackerLab 命令式 SuperMimic + AMP（当前主线） | `scripts/train.py` | MuJoCo-Warp，`third_party/trackerlab_supermimic` 的 `train_terrain.py` / `evaluate_terrain.py` / `export_command_policy.py` | `trackerlab_amp` |
| TrackerLab 命令式 SuperMimic + AMP（Isaac Lab / PhysX） | `python -m amp_supermimic.isaaclab_migration`，见[迁移与运行说明](isaaclab_migration/README.md) | 首版代码已实现，CPU 检查通过，GPU 短测待进行 | `himloco`（已验证 CPU 检查） |

> 所有 **HIM + AMP + SuperMimic** 的训练（包括原 HIM-PPO + AMP 基线 `train_amp_hie.py` / `launch_amp_hie.sh` 及其方案文档、smoke 输出）已迁至 [`../him_supermimic`](../him_supermimic/)，见 `../him_supermimic/docs/AMP_HIE_PLAN.md`。本目录只继续为其提供专家动作数据（`data/wblwbl/`）。

## 2. 目录与文件关系

```text
amp_supermimic/
├── configs/wblwbl_dog_v1.plan.json   机器人契约：关节顺序、dt、动作缩放、PD（来自 MyHIMLoco）
├── data/                              专家动作数据（见 §5、data/README.md）
│   ├── raw/                           原始下载素材（不改）
│   ├── go2/                           转成 Go2 CSV 的参考动作
│   ├── wblwbl/                        重定向到 WBLWBL 的动作（walk/turn/run/crawl/strafe/jump/…）
│   │   └── pools/<name>/              训练用专家池：train.json / test.json / manifest.json + CSV
│   ├── configs/                       go2_kine2go_*.json 等动作清单
│   └── MOTION_INVENTORY.md            某日盘点快照，非白名单
├── scripts/                           全部入口脚本（见 §3）
├── legacy/                           任务注册和本项目的环境、训练配置类
├── isaaclab_migration/               PhysX 入口、WMP 适配和测试
├── outputs/                           训练产物（trackerlab-amp/<run>/、预览视频）
├── OBSTACLE_DATA_PLAN.md              障碍动作数据审计
```

数据流：

```text
data/raw ──(prepare_* / retarget_go2.py / migrate_a1_wbl.py / retarget_lvr_reference.py)──▶ data/go2, data/wblwbl/<类别>/<clip>/{clip.csv,report.json,train.json}
data/wblwbl/<类别>/<clip> ──(build_style_pool.py)──▶ data/wblwbl/pools/<pool>/{train.json,test.json,manifest.json}
pools/<pool>/train.json ──(train.py train --amp-motion-config)──▶ outputs/trackerlab-amp/<run>/{iteration_*.pt,last.pt,run_config.json,train_log.csv}
checkpoint ──(train.py evaluate / play_mujoco.py / train.py export)──▶ 评估指标 / MuJoCo 回放 / actor.pt + policy.ts + metadata.json
```

## 3. 脚本一览（`scripts/`）

| 脚本 | 作用 |
| --- | --- |
| `train.py` | 主入口：`prepare` / `train` / `evaluate` / `export`；在内存中给 trackerlab 注册 WBLWBL 机器人、地形、步态，不改 third_party |
| `open_terrain.py` | `open_terrain` profile：3×2 共 6 级课程地形与命令奖励（猴子补丁） |
| `live_viewer.py` | 训练时的实时 MuJoCo 窗口（`--live-viewer`，CPU 子进程，显示速度箭头与 187 个高度扫描点） |
| `play_mujoco.py` | 加载 checkpoint 的闭环 MuJoCo 回放（CPU，支持 `--cycle` 自动轮播多种指令） |
| `play_motion.py` | 加载 `train.json` 动作池或单条 CSV 的 MuJoCo 参考动作轮播查看器（CPU） |
| `view_terrain.py` | 静态查看/截图地形 |
| `retarget_go2.py` | Go2（或 go2_flip_TO / NJU / A1-AMP JSON）→ WBLWBL 的 IK 重定向，并做 PD 回放审计，输出 CSV、`report.json`、`playback.npz`，通过筛选才写 `train.json` |
| `retarget_wblwbl_actions.py` | 批量重定向 `data/go2/kine2go_all40` + go2_flip_TO + NJU 跳跃到 `data/wblwbl/<walk/turn/run/strafe/crawl/jump>/`，写 `action_index.json`（参数 `--workers`） |
| `build_style_pool.py` | 把若干 `train.json` 片段打包成带 SHA256 的 `pools/<name>/`，并划分 train/test |
| `audit_wbl_gaits.py` | 用接触时序推断步态（trot/pace/bound/walk），输出标签 JSON（跳过 `rubbish` 目录） |
| `prepare_go2_motion_catalog_bundle.py` | 压缩整理 LVR HDF5 示范与 Go2 公开动作目录 |
| `prepare_kine2go_full_catalog.py` | 下载并校验固定版本的 40 条 Kine2Go 片段，转 Go2 CSV，写 `data/configs/go2_kine2go_all40.json` |
| `prepare_lvr_directions.py` | 从 LVR Isaac Lab 观测重建后退/侧移的根部位姿与关节角（需 `--repo --joint-order --output`） |
| `retarget_lvr_reference.py` | 用屈膝站姿把重建的 LVR 轨迹重定向到 WBLWBL（含左右镜像） |
| `migrate_a1_wbl.py` | A1 AMP 动作 + 程序化侧移 → WBLWBL 运动学候选（`--a1-mjcf --source --output`） |
| `prepare_obstacle_jump.py` | 下载 Kine2Go `solo8_jump_forward_b` 并重定向，作为"候选而非专家"（见 OBSTACLE_DATA_PLAN.md） |
| `render_*.py` | 离线 MP4 渲染（`render_wblwbl_catalog.py`、`render_go2_public_motion_catalog.py`、`render_kine2go_inward_review.py`、`render_a1_wbl.py`、`render_retarget.py`、`render_rear_comparison.py`），需要 `MUJOCO_GL=egl` 和 `ffmpeg` |

## 4. 训练、评估、导出

约定：

```bash
cd /home/qyw/supermimic
PY=/home/qyw/miniconda3/envs/trackerlab_amp/bin/python
```

`train.py` 全局参数：`--mjcf`、`--source-config`、`--profile {velocity,stairs,mountain,parkour,open_terrain}`、`--live-viewer`、`--viewer-envs`、`--viewer-env`、`--viewer-fps`、`--no-viewer-scan`、`--reset-warm-start-goal-enc`。`train` 之后的参数原样转给 `train_terrain.py`，任务 ID 自动设为 `locomotion/wblwbl_dog_v1/<profile>`。

```bash
# 训练（示例参数来自已有 run_config.json：4096 环境、AMP 池、reference/full 命令范围）
$PY amp_supermimic/scripts/train.py --profile open_terrain train \
  --run-dir amp_supermimic/outputs/trackerlab-amp/my_run \
  --num-envs 4096 --iterations 16000 \
  --amp-motion-config amp_supermimic/data/wblwbl/pools/turn_walk_trot5/train.json \
  --command-range-mode full

# 热启动 / 续训：--warm-start <ckpt>（新任务初始化）或 --resume <last.pt>
# 带实时窗口：在 --profile 后加 --live-viewer
# 评估
$PY amp_supermimic/scripts/train.py --profile open_terrain evaluate \
  --checkpoint <ckpt.pt> --output <dir> [--num-envs N --steps N --seed N]
# 导出策略（仅 actor）
$PY amp_supermimic/scripts/train.py export --checkpoint <ckpt.pt> --output <dir>
# 从回放生成参考（prepare）：--replay <回放> --output <目录>，写出 train.json/test.json
```

`train_terrain.py` 的完整参数（`--amp-weight`、`--buffer-steps`、`--ckpt-every` 等）见 `third_party/trackerlab_supermimic/scripts/rls/supermimic/train_terrain.py` 的 `parse_args`。
检查点每 `--ckpt-every`（默认 500）轮保存，`last.pt` 为最新。

**地形高程扫描范围**：`train.py` 的 `register()` 会通过 `_patch_height_scan()` 将 `HeightField` 覆盖为四足适配范围 **`X=[-0.40, 0.90] m`、`Y=[-0.38, 0.38] m`**（17×11 = 187 点，世界模型与 Critic 子采样 9×6 = 54 点），覆盖前后足落点与前方预判；训练、评估与 `play_mujoco.py` / `view_terrain.py` 共享该设置。注：此补丁生效前训练的历史 checkpoint（如 `wblwbl_open_flat_10k`）仍是按 TrackerLab 默认人型缩放范围训得的。

**导出产物**（`export_command_policy.py`）：`policy.ts`（TorchScript）、`actor.pt`、`metadata.json`。输入：观测历史 `[batch, hist, obs_dim]`（当前契约 `obs_dim=232`，含 187 点 17×11 高度扫描）和命令 `[batch, 12]`；动作公式 `clip(default_q + joint_offset + clip(action,-limit,limit)*action_scale, 关节限位)`。部署需要局部高度图和机体速度估计（见 metadata 的 `deployment_requirement`）。与 `robocon2027_sim_control/deploy_cpp` 的对接尚未在本文档验证。

**可视化**（一条命令即可连续轮播查看全部动作，无需逐条输入）：

```bash
# 1) 查看训练好的策略：自动轮播 前进 / 侧移 / 后退 / 转向 / 停止（--cycle）
/home/qyw/miniconda3/envs/trackerlab_amp/bin/python amp_supermimic/scripts/play_mujoco.py \
  --checkpoint <ckpt.pt> --level 0 --cycle

#    如需固定单一速度或自定义地形/扫描范围：
#    --vx 0.5 --vy 0.0 --wz 0.0        # 手动指定目标速度 (m/s, m/s, rad/s)
#    --level 0..5 --hfield             # 切换地形难度（Level 0 默认平面，--hfield 强制网格）
#    --speed 1.0 --no-follow --no-scan # 倍速、取消相机跟随、隐藏 187 个高程点
#    --quadruped-scan --apply-scan     # 覆盖预览并喂给策略；也可用 --scan-x X0 X1 --scan-y Y0 Y1

# 2) 查看 AMP 专家动作池：一条命令依次轮播 train.json 里的全部参考动作
/home/qyw/miniconda3/envs/trackerlab_amp/bin/python amp_supermimic/scripts/play_motion.py \
  --motion-config amp_supermimic/data/flat_trainning/pool/train.json

#    可选参数：
#    --clip forward                    # 只播文件名包含指定子串的动作
#    --csv <path/to/clip.csv>          # 只看单条 19 列 CSV 动作
#    --speed 0.5 --pause 0.8 --once    # 0.5 倍速慢放、片段间停顿、播完一遍退出

# 3) 静态查看地形与多视角截图
/home/qyw/miniconda3/envs/trackerlab_amp/bin/python amp_supermimic/scripts/view_terrain.py \
  --profile open_terrain --level 5     # 加 --snapshot-dir <目录> 保存 PNG
```

### AMP-SuperMimic 跑酷与梅花桩专项

在 `legacy/wblwbl_dog_v1/wblwbl_dog_v1_wmp_parkour_config.py` 中通过继承管理地形配置与奖励权重：
- `amp_supermimic_parkour`：包含 `flat_gap`、`tire_array`、`slope`、`stair` 四类地形（L0~L9，已移除梅花桩）。
- `amp_supermimic_stepping_stones`：继承自 `amp_supermimic_parkour`，专门训练 ROBOCON 梅花桩（L0~L14，L10 对齐规则标准尺寸），并配置梅花桩专项足端落点、防踩坑、摆动净空与防滑移奖励。

```bash
# MuJoCo 中查看跑酷 L9 与梅花桩 L10
$PY amp_supermimic/scripts/view_terrain.py --profile amp_supermimic_parkour --level 9
$PY amp_supermimic/scripts/view_terrain.py --profile amp_supermimic_stepping_stones --level 10

# 开始 MuJoCo-Warp + SuperMimic AMP 训练
$PY amp_supermimic/scripts/train.py --task amp_supermimic_parkour train
$PY amp_supermimic/scripts/train.py --task amp_supermimic_stepping_stones train

# 训练后在梅花桩地形回放
$PY amp_supermimic/scripts/play_mujoco.py --checkpoint <last.pt> \
  --profile amp_supermimic_stepping_stones --level 10
```

训练、地形预览和策略回放共用同一个配置。上坡和下坡各用一条恒坡度的光滑平面，`slope_grades` 按高差除以水平距离设置。默认用 2 cm 的网格采样地形，轮胎中心保留孔洞，桩和缺口的边缘按网格三角形插值。

## 5. 专家数据生成

### 5.1 格式契约
- 单文件：50 Hz CSV，19 列 = 根位置 `x,y,z` + 四元数 `qx,qy,qz,qw`（xyzw）+ 12 个关节角。
- 清单 `train.json`：`version, root, source_fps, quaternion_order, joint_names, files`。WBLWBL 清单的 `joint_names` 为 `RR,RL,FR,FL` 顺序，且 `source_fps=50`（`build_style_pool.py` 强制检查）；Go2 清单顺序为 `FL,FR,RL,RR`，Kine2Go 为 60 Hz。
- **永久规则**：`data/wblwbl/rubbish/**` 一律不得使用，详见 [data/README.md](data/README.md)。

### 5.2 本目录内已有的生成流程
```bash
# 1) 获取 Kine2Go 40 条 → data/go2/kine2go_all40
$PY amp_supermimic/scripts/prepare_kine2go_full_catalog.py
# 2) 批量重定向到 WBLWBL（调用 retarget_go2.py）
$PY amp_supermimic/scripts/retarget_wblwbl_actions.py --workers 4
# 3) 单条重定向
$PY amp_supermimic/scripts/retarget_go2.py --source <csv> --output <dir> --rotate-foot-calibration --swing-boost 0
#    其它格式：--optimized-json / --nju-json / --amp-a1-urdf；--neutral-foot-calibration --align-flat-ground
# 4) 步态标注
$PY amp_supermimic/scripts/audit_wbl_gaits.py <data目录> <mjcf> <输出json>
# 5) 打包训练池（默认类别 walk+turn，留出 walk/vhdc_horse1_s1_walk_03）
$PY amp_supermimic/scripts/build_style_pool.py --output amp_supermimic/data/wblwbl/pools/<name> \
  --clips turn/atomic_turn_left turn/atomic_turn_right walk/ai4_dog_walk_03 [--heldout …] [--style …] [--overwrite]
```
后退/侧移/障碍等候选的来源与状态见 `data/wblwbl/direction_candidates/`、`a1_candidates/README.md`、`obstacle_candidates/`，均为**运动学候选**，未经闭环物理验证。

### 5.3 使用 `third_party/supermimic_quadruped`
该仓库独立使用 `uv`（Python ≥3.12，<3.13，依赖 mjlab、mujoco-warp），本机 `which uv` 未找到，需先自行安装。其动作流水线（均在该目录下 `uv run python …`）：

1. `scripts/data/fetch_sources.py`：下载 AI4Animation/Tencent/CyberDog2 的 BVH 到 `data/raw/`。
2. 新增 `configs/robots/<name>.yaml`（参照 `go1.yaml`；要求 12 自由度、abad 绕 +x、thigh/calf 绕 +y、足端 site 位于 calf 关节正下方，`stance` 等字段见 `docs/PORTING.zh-CN.md`）。**WBLWBL 的 MJCF 是否满足该结构、有无足端 site，尚未验证**（MJCF 中仅检索到 `imu` 一个 site，足端是 calf body 上的 sphere geom，需要自行添加足端 site 并检查几何约束）。
3. `scripts/data/build_dataset.py --robot <name>` → `data/npz/<name>/*.npz`、`manifest.csv`、`clips.yaml`、`report.md`；`check_library.py --robot <name>` 校验。
4. `screen_clips.py` / `assign_groups.py` 需要该机器人的训练 spec 与 GPU。

**格式差异**：该库输出 mjlab 风格 `.npz`（`joint_pos, joint_vel, body_pos_w, body_quat_w(wxyz), …`，关节按 MJCF 顺序），而本目录的 TrackerLab 链路读 19 列 CSV。仓库内**没有**现成的 npz→CSV 转换脚本；若要桥接，需按 `body_pos_w/body_quat_w` 的根体取 `x,y,z` 与 `quat(wxyz→xyzw)` 再拼 `joint_pos`（按 `RR,RL,FR,FL` 重排）并自写清单，需要时再实现并校验。

## 6. 第三方库用途

| 目录 | 用途 | 本工程中的使用 |
| --- | --- | --- |
| `trackerlab_supermimic` | TrackerLab SuperMimic：世界模型 + AMP 命令训练、评估、导出、MuJoCo-Warp 地形 | 本目录训练的核心 |
| `supermimic_quadruped` | 四足 SuperMimic + dog mocap 重定向库 + 摇杆规划器 | 可选的动作生成方案（§5.3） |
| `MyHIMLoco` | HIMLoco 训练代码与 WBLWBL 的 Isaac Lab 配置 | `configs/wblwbl_dog_v1.plan.json` 的参考；PhysX 入口复用 WBLWBL 的 URDF 与网格，运行参数由本项目 `legacy` 配置提供；HIM 训练在 `../him_supermimic` |
| `AMP` | Isaac Lab 的 A1 AMP 复现 | 本目录不直接使用（`../him_supermimic/scripts/train_amp_hie.py` 载入其判别器） |
| `Latent-Variation-Regularization` | Go2 前进/后退/侧移示范（ICRA 2026） | `prepare_lvr_directions.py` 的数据来源 |
| `go2_flip_TO` | Go2 轨迹优化与 50 Hz 动作数据 | `retarget_wblwbl_actions.py` 读取 `datasets/go2/mocap_motions_go2/*_50hz.txt` |
| `go2_supermimic` | Go2 追逐/躲避任务（mjlab） | 仅参考 |
| `supermimic_official` | SuperMimic 官方（G1 人形，uv + mjlab） | 仅参考 |

机器人资产（MJCF/URDF）取自 `/home/qyw/robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/`，不在本仓库内。

## 7. 注意事项
- `train.py` 默认读取 `../him_supermimic/outputs/zero-start/b-seed511/config.json` 作为 `--source-config`。
- `outputs/trackerlab-amp/wblwbl_open_trot5_*` 等是实验产物，单个 checkpoint 约 200 MB，勿提交。
- 渲染脚本需要 EGL；`play_mujoco.py` 在 Level 0 默认使用平面地面以避免 CPU 高度场接触问题。
