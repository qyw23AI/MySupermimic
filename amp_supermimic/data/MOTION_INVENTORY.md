# 本地动作数据分类与缺口（2026-11-15）

> 本清单是当日盘点快照，不是当前训练白名单。当前目录排除和准入规则以 [README.md](README.md) 为准。尤其必须跳过 `wblwbl/rubbish/**` 的所有内容。下方关于 WBLWBL 条数和候选状态的描述仅记录该快照时的状态。

仓库：`/home/qyw/supermimic`。按**独立来源轨迹**计数，不把原始下载、转换 CSV、裁剪片段、压缩 HDF5 和视频重复计数。当前快照先记录补充前状态；下方另记本轮新增内容。

## 1. 按来源与处理阶段

| 来源 | 已有独立轨迹 | 位置 | 数据性质与可用性 |
|---|---:|---|---|
| Kine2Go | 40 | `amp_supermimic/data/raw/kine2go_all40/`，转换版 `data/go2/kine2go_all40/` | 原始片段及逐片段元数据；Go2 版 60 Hz 根部位姿、四元数、12 关节角 CSV。是运动学参考，未逐条通过 Go2 动力学回放。 |
| LVR | 6 = 3 方向×仿真/真机 | `data/raw/go2_lvr_backward_sideways/`，整理版 `data/go2/lvr_velocity_demos/` | Go2 前进、后退、侧移观测/动作 HDF5；仿真原始文件预分配 100 万行、有效前 1000 行，另有压缩副本。缺全局根部位姿，不能直接作为全身 AMP 轨迹。 |
| go2_flip_TO | 5 | `third_party/go2_flip_TO/datasets/go2/mocap_motions_go2/` | 50 Hz、49 维 Go2 优化生成轨迹；非实机示范。 |
| WBLWBL 重定向 | 1 | `data/wblwbl/ai4_dog_trot_00.csv` | 从 Kine2Go trot 转换出的运动学候选，不能再计为第 52 条独立来源动作。关节限位与足端拟合通过，但 PD 物理回放的前进进度未通过。 |

以上 `data/` 路径均相对 `amp_supermimic/`；`third_party/` 路径相对仓库根。

## 2. 40 条 Kine2Go 按物种、动作和来源细分

| 原始来源 | 类别 | 条数 | 文件名（不含扩展名） |
|---|---|---:|---|
| AI4 狗自然动作 | 慢走 | 6 | `ai4_dog_walk_00, 01, 02, 03, 04, 06`（均加共同前缀） |
| AI4 狗自然动作 | 小跑 | 2 | `ai4_dog_trot_00, ai4_dog_trot_01` |
| AI4 狗自然动作 | 跑 | 3 | `ai4_dog_run_00, ai4_dog_run_02, ai4_dog_run_04` |
| AI4 狗自然动作 | 侧对步、慢跑 | 2 | `ai4_dog_pace, ai4_dog_canter` |
| AI4 狗自然动作 | 左转、右转 | 2 | `ai4_dog_left_turn, ai4_dog_right_turn` |
| AI4 合成狗动作 | 圆形、8 字、椭圆、方形路线行走 | 4 | `ai4_dog_synth_circle_walk, ai4_dog_synth_eight_walk, ai4_dog_synth_ellipse_walk, ai4_dog_synth_square_walk` |
| AI4 合成狗动作 | 小幅、大幅侧移 | 2 | `ai4_dog_synth_tight_strafe, ai4_dog_synth_wide_strafe` |
| AI4 合成狗动作 | 半翻转跳跃 | 1 | `ai4_dog_synth_half_flip_jump` |
| Solo8 机器人 | 快/慢爬行 | 2 | `solo8_crawl_fast, solo8_crawl_slow` |
| Solo8 机器人 | 两种向前跳 | 2 | `solo8_jump_forward_a, solo8_jump_forward_b` |
| Solo8 机器人 | 前蹭、走 | 2 | `solo8_scoot_forward, solo8_walk` |
| VHDC 马 | 两次采集，每次 3 段走和 3 段小跑 | 12 | `vhdc_horse1_s1_walk_01..03, s1_trot_01..03, s2_walk_01..03, s2_trot_01..03`（每项共享 `vhdc_horse1_` 前缀） |

合计狗 22、Solo8 6、马 12。每个原始片段有 `motion.npy`、`clip.json`、`config.json`；逐条时长、许可和映射参见 `data/go2/MOTION_CATALOG.md` 与 `data/go2/kine2go_all40/manifest.json`。AI4 狗为 CC BY-NC 4.0，马为 CC BY-SA 4.0，Solo8 为 BSD-3-Clause（以逐片段元数据为准）。

## 3. 5 条 Go2 轨迹优化动作

| 类别 | 文件 |
|---|---|
| 后空翻 | `quad_backflip_50hz.txt` |
| 前空翻 | `quad_frontflip_50hz.txt` |
| 约 1 m 前跳 | `quad_jump_forward_1m_50hz.txt` |
| 左/右侧翻 | `quad_sideflip_50hz.txt`, `quad_sideflip_right_50hz.txt` |

仓库里的 `quad_walk_backward_ramp.py`、`quad_walk_forward.py`、`quad_strafe_left_ramp.py`、`quad_strafe_right_ramp.py`、`quad_spin_inplace.py`、`quad_spin_inplace_reverse.py` 是求解脚本，**不是已经生成的动作数据**；无对应命名的全身轨迹文件。`output_go2.txt` 未标明对应上述哪种动作，暂不归类为已证实的后退/转向。

## 4. 6 组 LVR 观测示范

| 方向 | Isaac Lab 仿真 | Go2 真机 | 可否恢复全局运动轨迹 |
|---|---|---|---|
| 前进 | `isaac_go2_forward_traj-1.hdf5` | `unitree_go2_forward_step-250+250.hdf5` | 否，缺全局根部位姿 |
| 后退 | `isaac_go2_backward_traj-1.hdf5` | `unitree_go2_backward_step-250+250.hdf5` | 否，缺全局根部位姿 |
| 侧移 | `isaac_go2_sideway_traj-1.hdf5` | `unitree_go2_sideway_step-250+250.hdf5` | 否，缺全局根部位姿 |

因此“完全没有后退动作”不成立；准确表述是**已有后退的仿真与真机观测/动作示范，但尚无经验证的 WBLWBL 全身后退专家轨迹**。

## 5. 重复关系和训练门槛

- `raw/kine2go_full_ai4/` 与 `go2/kine2go_full_ai4/` 是全部 40 条中的 AI4 狗 22 条子集；`raw/kine2go_inward_review/`、`go2/inward_review/` 为 7 条评审子集及裁剪片段；`raw/kine2go/` 和 `go2/ai4_dog_trot_00.csv` 是同一 trot 的早期副本。不要累计成新动作。
- `lvr_velocity_demos/` 里的 6 份是原始 LVR 的有效区间压缩/复制版本；三份大型仿真原始 HDF5 不能按预分配容量推算时长。
- `outputs/go2_motion_catalog/` 的视频与打包文件只是 40+5 条的预览和分发包，不能算新增示范。
- WBLWBL 当前只有 `ai4_dog_trot_00` 重定向候选。`data/wblwbl/report.json` 标出 `motion_usable_for_amp=true`（运动学筛选），同时 `playback_pass=false`、`playback_progress_ratio=-0.02112`；不能把它表述为已通过物理跟踪的专家动作。

## 6. 对照公开来源的缺口与处理

| 候选 | 仓库当前状态 | 本轮处理 |
|---|---|---|
| Kine2Go 40 条、LVR 6 组、go2_flip_TO 5 条 | 都已下载 | 跳过重复下载 |
| go2_flip_TO 的后退、侧移、原地旋转 | 仅有生成脚本，没有生成结果；本机现用环境缺求解依赖 | 明确保留为待生成；不拿 LVR 观测伪装为全身轨迹 |
| NJU-RLC quadrupedal-agility 的跨障碍跳跃片段 | 补充前尚未下载；与已有普通前跳/翻转是不同片段 | 已下载 `jump_0, jump_1, jump_3, jump_4` 原始 JSON；仍需映射/动力学验证 |
| 完整 Quad Imaginarium 数据集 | 官方称暂未发布 | 无可下载完整集 |
| WBLWBL 可物理跟踪专家后退、侧移、跳跃 | 当前缺失 | 需专门重定向及回放筛选，不能靠复制 Go2 文件解决 |

## 7. 本轮新增数据（独立来源片段 4 条）

`data/raw/nju_agility_jump/` 收录 `jump_0.json`（31 帧）、`jump_1.json`（33 帧）、`jump_3.json`（33 帧）、`jump_4.json`（33 帧），帧间隔均为 1/30 s，每帧 61 个数。`SOURCE.json` 记录项目、原始相对路径和格式；`LICENSE` 为上游仓库 MIT 许可证。未修改原始数值。它们是原始跨障碍动作素材，尚未证明可直接用于 Go2 或 WBLWBL AMP；尤其不能把 61 维 JSON 当成已有的 19 列 Go2 CSV。补充后独立来源片段按文件计为 Kine2Go 40 + LVR 6 + 优化轨迹 5 + NJU 跳跃 4 = 55；WBLWBL 的 1 条重定向仍是派生副本。

本文件分类依据本地真实文件、manifest、源 README 与 WBLWBL 报告。
