# WBLWBL Dog V1 赛道参考动作

这两条轨迹用 WBLWBL Dog V1 的 MuJoCo 碰撞模型和足球半径求解。Z 高度随地形变化，CSV 与对应障碍坐标同时定义。赛道模型中的单木桥是 `platform_04`，高墙是 `high_wall_00`。

| 动作 | 障碍几何与局部坐标 | 序列 |
| --- | --- | --- |
| `single_bridge_trot` | 桥面从 X=0 到 X=3 m，Y=±0.20 m，顶面 Z=0.10 m | 0.32 m/s，足端中心间距 0.28 m。RR–FL 与 RL–FR 交替，包含桥前、上桥、桥上与完全下桥 |
| `high_wall_climb` | 墙沿 X 厚 0.08 m，Y 宽 0.80 m，顶面 Z=0.30 m，墙体中心 X=0、Y=0 | 前脚依次搭顶，后脚收近，前脚落到远侧，后脚依次上墙并落地，最后恢复站姿 |

每个子目录都含 50 Hz 的 19 列姿态 CSV、TrackerLab 格式 `train.json`、接触计划 `support.csv`、时段和校验指标 `report.json`、双视角 `reference.mp4`。姿态为根部 XYZ、XYZW 四元数、RR/RL/FR/FL 各腿 hip/thigh/calf 关节角，单位为米和弧度。接触计划按 RR/RL/FR/FL 排列，0 为摆动、1 为地面支撑、2 为障碍顶面支撑。CSV 含真实障碍高度，做参考重置时应搭配同尺寸地形。

回放逐帧设置模型位姿，不使用 PD 或策略。用 200 Hz 插值检查模型碰撞几何、关节限位和每脚落点。高墙轨迹的静态质心投影在部分单腿摆动时段超出支撑多边形，最低有符号余量约 −11 cm。这个量只描述几何投影，不能反映不同高度足端间的接触力分配。使用前可先做动力学跟踪，验证足端支撑、滑移、倒地和关节力矩。

重新生成数据或视频：

```bash
cd /home/qyw/supermimic
/home/qyw/miniconda3/envs/trackerlab_amp/bin/python -B \
  amp_supermimic/scripts/generate_obstacle_gaits.py --render
```

用 `--kind bridge` 或 `--kind wall` 只生成一条，用 `--output` 指定另一个输出目录。
