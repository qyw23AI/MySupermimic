# Go2 内收步态候选动作预览

## 内容

- `selected_clips/`：按 Kine2Go 固定版 `motion_ranges.txt` 截取的 7 段 Go2 参考轨迹，CSV 为根位置、根四元数（xyzw）和 12 个关节角。
- `../../raw/kine2go_inward_review/`：原始 Kine2Go `motion.npy`、每段元数据、许可证和固定版帧区间清单。
- 视频：`../../../outputs/go2_inward_review/go2_expert_inward_candidates.mp4`。
- `../../../outputs/go2_inward_review/inward_candidate_metrics.json`：足端内收程度的辅助统计。

视频直接在 Go2 MJCF 中逐帧显示参考姿态。两侧相机的方位角固定在世界坐标系中，镜头中心跟随机器人位置，因此根部 yaw 转动会清楚可见；它不是 PD 动力学回放，也不是训练出的策略。

## 候选片段

| 动作 | 上游帧区间（含首尾） | 片段时长 | 足端横向距离 P05 |
|---|---:|---:|---:|
| 普通步行 00 | 101–580 | 8.00 秒 | 8.45 厘米 |
| 普通步行 01 | 377–1050 | 11.22 秒 | 7.79 厘米 |
| Pace | 162–201 | 0.65 秒，预览重复 4 次 | 9.55 厘米 |
| 左转 | 1085–1124 | 0.65 秒，预览重复 4 次 | 3.71 厘米 |
| 右转 | 2404–2450 | 0.77 秒，预览重复 4 次 | 8.02 厘米 |
| 圆周行走 | 446–806 | 6.00 秒 | 6.51 厘米 |
| 8 字行走 | 全片段 | 19.58 秒 | 3.90 厘米 |

P05 是所有足端相对机身横向距离的第 5 百分位；按该指标，越小表示片段中足端越常靠近身体中线。它只是筛选线索，最终以视频和足端轨迹为准。以上片段没有出现足端越过身体中线的情况。

## 来源与许可证

参考轨迹来自 MIMUW-Robotics/Kine2Go 固定版本；上游动作区间来自对应 pipeline commit 的 `motion_ranges.txt`。这些 `ai4_dog_*` 动作依据 AI4Animation 狗运动捕捉数据生成，许可证为 **CC BY-NC 4.0**。引用时保留原始署名：H. Zhang et al., “Mode-Adaptive Neural Networks for Quadruped Motion Control,” SIGGRAPH 2018。许可证文本和逐段元数据都保存在 raw 目录中。

原始 NPY 有时包含比单个动作更长的序列，因此 `selected_clips/` 已按上游帧区间截取；原始文件保持完整。
