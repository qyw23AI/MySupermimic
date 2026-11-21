# WBLWBL Dog V1 双阶段跳墙参考

场景是赛道高墙 `high_wall_00` 的局部坐标复现。墙体沿 X 厚 0.08 m、沿 Y 宽 0.80 m、顶面 Z=0.30 m，位于 X=0、Y=0；机器人沿 +X 越墙。

两条参考共用前半段。先后腿蓄力，前双脚离地并搭上墙顶；随后前双脚保留墙顶支撑，后双脚往墙边收，再进行第二次蹬跳。参考的两次全腾空段分别为 0.38–0.58 s 和 1.48–1.80 s，计划前双足同时在墙顶支撑 0.74 s。

- **直越版** `high_wall_jump_direct`，2.36 s。第二跳后，前双脚先落到远侧，后双脚越过墙顶并跟随落地。该路线只要求前脚踩准 8 cm 厚的墙顶。
- **后双足踩顶版** `high_wall_jump_rear_touch`，2.78 s。第二跳结束时前双足落地，后双足同步踩上墙顶，保留约 0.48 s 计划支撑后下墙。后足球外缘距离墙顶最近边缘约 6.2 mm，适合用于对比触顶动作，对落脚误差更敏感。

各目录内的 `reference.mp4` 为无 PD 的运动学回放。姿态 CSV 为 50 Hz、19 列，依次是根部 XYZ（米）、XYZW 四元数、RR/RL/FR/FL 四条腿的 hip/thigh/calf（弧度）。`support.csv` 按 RR/RL/FR/FL 排列，0 为摆动或腾空、1 为地面支撑、2 为墙顶支撑。`report.json` 记录相位、碰撞检查与动力学诊断。`train.json` 可作为独立的 TrackerLab 参考清单。

两条轨迹通过关节限位和 200 Hz 插值碰撞检查。自由机身诊断只施加关节 PD 力矩，力矩按模型的 ±23.7 N·m 裁剪，测试了四组增益。所有尝试的前双足同时在墙顶支撑时间均为 0，均未完成越墙。第一跳需要实际学到后腿推地和前抬分工，参考中的根部弹道本身不会给自由机身施加起飞冲量。闭环策略验收可按顺序检查前双脚墙顶支撑、第二次腾空、全部足球越过远侧墙沿及稳定着地。

```bash
cd /home/qyw/supermimic
/home/qyw/miniconda3/envs/trackerlab_amp/bin/python -B \
  amp_supermimic/scripts/generate_wall_jump.py --render --physics-check
```
