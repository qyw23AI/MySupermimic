# WBLWBL Dog V1 单木桥 trot 与高墙攀爬

两项任务复用 WBL 的 PhysX、TrackerLab SuperMimic 训练器和 WMP 的赛道网格、接触记录、难度课程。各自热启动平地 Actor，世界模型和 Critic 针对新地形重新训练。两个检查点分别保存。

| 任务 | 课程 | 固定评估 |
| --- | --- | --- |
| `bridge_trot` | 3 m 长，桥宽从 70 cm 缩到 40 cm，桥高从 5 cm 升到 10 cm；从短边前起步 | L9，0.32 m/s。从另一端下桥后保持稳定；桥中段剔除侧落和两脚触地，分别检查 RR–FL 与 RL–FR 的实际支撑占比和至少四次交替 |
| `high_wall` | 墙宽 80 cm、沿前进方向厚 8 cm，高度 5、8、12、15、20、25、30 cm | L6，0.45 m/s。统计全身越墙稳定落地与墙顶前脚支撑；严格成功额外要求至少 80 ms 的有效墙顶前脚支撑 |

桥的 AMP 输入是 `data/wblwbl/narrow_bridge_trot/train.json`，三条参考的足中心间距是 26、28、30 cm，速度是 0.26、0.32、0.38 m/s。生成脚本使用对角相位及 WBL 足端逆运动学，视频直接回放关节与根位姿。高墙只对入场步态使用低权重 AMP，墙顶的接触奖励来自真实仿真，想象阶段使用可微的中心线、抬身和前脚对准墙顶的引导。窄桥在想象阶段也惩罚足球外缘超出桥面。8 cm 窄墙面有可能出现跳越或触顶后跃下，严格成功需结合视频检查实际动作。

```bash
cd /home/qyw/supermimic
bash amp_supermimic/scripts/run_wbl_obstacle_policies.sh all
```

将 `all` 换为 `bridge_trot` 或 `high_wall` 可单跑其中一项。脚本默认每项 512 环境和 6000 轮，按课程进度逐级训练，保存常规检查点并在比赛尺寸评估 `last.pt`。运行目录包含训练日志、课程尝试和成功计数。评估记录完整回合数、跌倒、超时、普通与严格成功率，可用 `WBL_NUM_ENVS`、`WBL_ITERATIONS` 和 `WBL_WARM_START` 覆盖规模和热启动权重。当前课程状态没有接入恢复流程，正式任务中断后先保留原目录。

## SuperMimic-Quadruped 的用途

[SuperMimic-Quadruped](https://github.com/Houserz/supermimic-quadruped) 的 `quadmotion` 能在给定目标资产、关节映射及足端几何后，重定向来源动作并做足锁定与筛查。Rover X1 的公开规划模型接收摇杆速度和周期相位，地形高度和桥边位置需要另接规划器或改用感知策略。训练它的跟踪器还要补 WBL 机器人规格、地形碰撞和任务终止。当前使用项目里已经打通的 WBL TrackerLab/PhysX 链路，便于把策略、机器人模型和赛道评估对齐。

## 短测

首版热启动各执行 4 环境、3 轮更新，并用种子 801 在比赛尺寸下评估 4 个完整回合。桥和墙均为 0/4 成功、4/4 超时。前进速度 RMSE 分别为 0.316 m/s（指令 0.32 m/s）和 0.445 m/s（指令 0.45 m/s），当前热启动在 PhysX 的前进跟踪很弱。随后加入可微足端奖励和持续接触判定，再次验证训练更新和保存检查点。短测产物仍需要正式训练和重新评估。
