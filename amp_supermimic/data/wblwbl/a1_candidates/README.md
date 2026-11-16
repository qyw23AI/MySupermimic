# A1 → WBLWBL 动作审查

这个目录包含从本地 MyAMP 完整迁移的 15 条 A1 运动学动作，以及 2 条程序生成的 WBL 侧移候选。每个子目录内的 CSV 为 50 Hz，依次是世界根部位置、XYZW 四元数和 12 个 WBL 关节角。report.json 记录顺序、文件哈希、运动方向、足底高度和关节检查。reference.mp4 仅播放运动学轨迹。A1 来源的动作左侧是 A1、右侧是 WBL，视频以 0.5 倍速播放。侧移视频按原速播放。

## 来源

- 本地 MyAMP 的 mocap_motions_a1 提供 61 维动作。A1 关节及脚点都按 FL、FR、RL、RR 排列。逐帧用 A1 模型核对源关节角的正运动学脚点，与文件内保存的脚点进行比较，再按 WBL 的 RR、RL、FR、FL 排列。
- [motion_imitation 原始后退动作](https://github.com/erwincoumans/motion_imitation/tree/d0e7b963c5a301984352d25a3ee0820266fa4218/motion_imitation/data/motions) 的两条 19 维后退文件与本地保存文件逐字节一致。对应前进动作的姿态字段是精确时间倒序。本次使用 MyAMP 中从同源关键点重新生成、带足端的 61 维版本 pace_reverse0 和 trot_reverse0，避免重复纳入同源动作。WBL 参考放慢 1.6 倍，水平位移缩放为 0.85。对照视频中的 A1 与 WBL 使用相同时间轴。
- A1 预览模型来自 [MuJoCo Menagerie 的 Unitree A1](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_a1)。本地原始素材和 AI4Animation 使用声明位于 ../../raw/a1_amp_candidates。
- 本地 15 条 A1 文件没有纯侧移。两条 procedural_strafe 分别用 +0.33 和 -0.33 m/s 的根部速度、对角足相位和固定世界坐标支撑脚生成，是独立的 WBL 运动学候选。

## 复现

从 /home/qyw/supermimic 运行以下命令。A1 模型路径替换成上述固定版本的本机 Menagerie 目录。

```bash
PY=/home/qyw/miniconda3/envs/trackerlab_amp/bin/python
A1=/你本地的/mujoco_menagerie/unitree_a1
OUT=/tmp/a1_wbl_review
"$PY" -B amp_supermimic/scripts/migrate_a1_wbl.py \
  --a1-mjcf "$A1/a1.xml" \
  --source amp_supermimic/data/raw/a1_amp_candidates/mocap_motions_a1 \
  --output "$OUT"
"$PY" -B amp_supermimic/scripts/render_a1_wbl.py "$A1/scene.xml" "$OUT"/*/
```

静态审核覆盖足底穿地、关节限位、关节速度和目标足端误差。后退轨迹短，首尾存在关节接缝，重复采样时应按独立片段处理。当前为运动学参考候选，尚无闭环跟踪结果。
