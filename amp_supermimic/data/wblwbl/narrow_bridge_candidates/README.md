# WBLWBL Dog V1 单木桥前进参考

赛事 V2.0 的单木桥为 3.00 m × 0.40 m × 0.10 m。MuJoCo 赛道中的对应几何体为 `platform_04`。这里生成的是**四脚已经上桥后的直线前进段**，桥面中心线为世界坐标 X 轴。上桥和下桥所需的抬脚、俯仰及高度变化需要另外采集。

| 动作 | 左右足中心间距 | 名义前进速度 | 足球外缘距侧边最小余量 |
| --- | ---: | ---: | ---: |
| `bridge_walk_w28_v22` | 28 cm | 0.22 m/s | 36.2 mm |
| `bridge_walk_w30_v28` | 30 cm | 0.28 m/s | 26.2 mm |
| `bridge_walk_w32_v32` | 32 cm | 0.32 m/s | 16.2 mm |

各条动作按 RR → FR → RL → FL 的顺序循环摆动，每只脚的支撑占比约 80%。支撑脚在世界坐标中固定，摆动脚起落用连续曲线。躯干平移按支撑多边形做周期调节，关节角由 WBL 实际模型上的四脚 IK 计算。左右脚没有跨越中心线，膝盖保持弯曲。报告使用模型里 23.8 mm 的球形足半径扣除桥边余量。

- `train.json` 可传给 TrackerLab 的 `--amp-motion-config`。清单只包含姿态 CSV，50 Hz、19 列、四元数 XYZW、关节顺序 RR/RL/FR/FL。
- `*_stance.csv` 是计划支撑相位标签，按 RR/RL/FR/FL 排列。现有 AMP 读取器从姿态计算特征。
- `report.json` 记录足球边距、足底净空、世界系支撑脚滑移、关节和静态质心支撑余量。MP4 为 WBL 在同尺寸桥上的双视角运动学回放。CSV 的支撑平面高度为 0，录像时整体抬升 10 cm。
- 这些是运动学候选样本。回放按帧设置根状态和关节角，实测过桥成功率仍需要闭环仿真；三个样本也不构成独立的策略评估。

从仓库根目录重新生成 CSV、报告和 MP4：

```bash
/home/qyw/miniconda3/envs/trackerlab_amp/bin/python -B amp_supermimic/scripts/generate_bridge_walk.py --render
```

## 已核对的来源

- [SuperMimic-Quadruped](https://github.com/Houserz/supermimic-quadruped) 提供犬只动捕重定向、动作筛选和动作跟踪流程。检查了本地版本的来源目录和默认目标机器人配置，未定位到带桥宽标注的内收前进轨迹。
- [Kine2Go](https://huggingface.co/datasets/MIMUW-Robotics/kine2go/tree/main/data) 发布 Go2 多类运动学轨迹。其目录未标出单木桥内收前进样本。
- [HyQ Line Walking](https://arxiv.org/abs/2007.01087) 和 [ANYmal Learning Agile Locomotion on Risky Terrains](https://arxiv.org/abs/2311.10484) 都演示了四足窄梁通过。HyQ 研究的是 6 cm 窄梁的双足支撑平衡，ANYmal 使用学习策略；未找到可直接作为本项目参考 CSV 的逐帧发布轨迹。
