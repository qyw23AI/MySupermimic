# WBL 在 SuperMimic-Quadruped 中跟踪双跳过墙

训练器来自仓库中的 `third_party/supermimic_quadruped`。本适配保留其真实仿真采样、世界模型和想象更新。训练场景使用 WBL 的碰撞几何、40/1 PD、23.7 N·m 力矩上限，以及 30 cm 高、8 cm 厚、80 cm 宽的墙。物理 200 Hz、策略 50 Hz。墙位于坐标原点，世界模型和价值网络额外读取水平位置与朝向。策略使用相对参考姿态作为目标。将固定墙换成随机墙体时，需要让观测和模型显式感知相应的几何变化。

`prepare` 使用上游 `quadmotion.clip.body_states_from_pose` 转换直越 CSV，导出跟踪 NPZ。第一跳从墙前到前足墙顶、后足收回后的姿态，第二跳训练从该姿态启动。完整阶段从墙前开始，整段跟踪直到两跳落地。训练可以从参考中的若干帧重置；评估统一从各阶段的第 0 帧开始，完整评估全程连续仿真。

## 运行

```bash
cd /home/qyw/supermimic
bash amp_supermimic/scripts/run_wall_jump_tracking.sh first
bash amp_supermimic/scripts/run_wall_jump_tracking.sh continue /绝对路径/第一阶段运行目录
```

`all` 可串联运行。第一和第二阶段都需有接触验收通过的回合，脚本再进入下一阶段。脚本使用锁定的 Python 3.12 环境，第一次安装下载 CUDA 组件。可设置 `WBL_PYTHON312` 指向本机 Python 3.12，或用 `WBL_SM_PY` 指定已有兼容环境。训练预算可用 `WBL_ITERS`、`WBL_ENVS`、`WBL_BATCH`、`WBL_HORIZON` 改写。初次默认 256 环境、6000 次采样迭代和 16 步想象；训练统计写入各阶段 `tb/` 和训练日志。

```bash
bash amp_supermimic/scripts/run_wall_jump_tracking.sh eval /绝对路径/运行目录 first
# 也可以把 first 换成 second 或 full，并在后面指定另一个 PT 检查点。
```

## 使用已通过接触验收的第一跳轨迹

设置 `WBL_FIRST_TEACHER` 指向一份包含仿真 `states` 和实际 `controls` 的 NPZ。两者分别是 37 维状态和 12 维关节位置目标，采样频率为 50 Hz，状态比控制多一帧。训练器在同一个 Warp 环境中从墙前复放该控制序列，并按接触与机身稳定条件重新验收。通过后保存 `ckpt_teacher.pt`，用动作监督预热 Actor。第 180 轮之前只做世界模型更新，之后用想象回报和示范动作误差共同更新 Actor。

```bash
WBL_FIRST_TEACHER=/绝对路径/first_teacher.npz \
  bash amp_supermimic/scripts/run_wall_jump_tracking.sh all /绝对路径/新的运行目录
```

第二跳训练由第一跳通过回合的终态初始化，通过短暂的调整段接入第二跳参考。完整评估仍从墙前启动，连续验证两个跳跃。通过第一跳的单独评估后，脚本会先记录实际交接状态。

## 验收与导出

- 第一跳要求实际腾空，随后前双足球同时落在墙顶可支撑区域，有真实法向接触力，连续支撑至少 0.10 s。整个连续支撑窗口内，后双足都在地面，机身速度和姿态保持在稳定区间。
- 第二跳要求从前双足支撑且后双足在地面的姿态再次腾空，四足都越过墙体远侧，四足落在地面稳定至少 0.10 s。测试中的机身、大小腿碰撞会取消通过。
- `evaluation.json` 记录实验通过数和检查点 SHA-256，`rollouts.npz` 是实际自由机身仿真的关节和根状态。只有完整阶段的 learned 模式通过全部次序后，才会从该评估回合写出 `amp/train.json` 和 AMP CSV。第一跳和第二跳的单独评估均不会导出整套专家数据。

评估中的 takeoff_diagnostics 记录双前足中较低一只的最高足底高度、该时刻的水平位置、机身峰值上升速度和后足峰值法向力。只有双前足足底达到墙顶附近时，才报告落点相对墙中心的偏差。

## 前小腿搭墙方案

以下实验从上一版闭环第一跳通过回合的实际姿态开始。足端组和小腿组使用相同起点和相同的 WBL MuJoCo 模型，包括 200 Hz 物理步长、50 Hz 位置指令、40/1 PD、23.7 N·m 力矩上限和原 30 cm 墙。

- `brace_experiment.py` 搜索一段开环关节位置序列。小腿组只放行两条前小腿接近垂直法向的墙顶接触。承重补步要求前侧两个支点和两个后足连续承重 0.10 s，机身前移超过 4.5 cm，每只后足前移超过 7.5 cm。之后双小腿需卸载 0.10 s，后足持续承重且保留前移。
- `brace_second.py` 在新仿真中复验卸载，再把第二跳接在同一段控制序列末尾。验收包括实际腾空、四足越过墙远侧、卸载后前小腿无回碰，以及落地四足承重连续 0.20 s，机身高度大于 22 cm。
- `brace_replay.py` 从相同的初始状态独立重放控制序列，输出接触与前移记录。 `brace_view.py` 使用同一套回放制作并排视频，在各自验收帧定格。

本次实验可这样复验。

```bash
cd /home/qyw/supermimic
base=/home/qyw/supermimic/amp_supermimic/outputs/wall_jump_tracking
src="$base/first-teacher-anchored-20261117-121208"
run="$base/calf-brace-20261117-124549"
"$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.brace_replay \
  --work "$src" --toe "$run/toe-control.npz" --calf "$run/calf-unload.npz" \
  --second "$run/calf-second.npz" --output "$run/brace-evaluation-recheck.json"
```

把模块名换成 `amp_supermimic.wall_jump_tracking.brace_view`，把输出后缀换成 `.mp4` 可以重新渲染。无显示器时可以设置 `MUJOCO_GL=egl`。

## 小腿路线的 SuperMimic 闭环

`calf_tracking.py` 把验收过的 MuJoCo 小腿路线转成 SuperMimic 跟踪片段。第二阶段从原 16 次扰动实验使用的墙前支撑态启动。完整阶段用第一跳成功回放的实际状态和关节目标接上小腿路线，从墙前地面启动，评估不在阶段之间重置。Actor 可用 `--guide` 保留参考动作时序并学习由观测和目标驱动的残差。`--landing-feedback 108` 将落地姿态和足底高度反馈纳入示范数据，推理时由 Actor 输出关节目标。仍由上游训练循环采集物理过渡、拟合世界模型并进行想象轨迹 BPTT。

`calf_refine.py` 使用训练专用的扰动起点修正教师。固定评估批次为原来的 seed 301、16 个起点、XY 各 ±2 mm、朝向 ±0.5°。`--strict-contact` 额外检查物理子步的接触峰值。AMP 导出要同时满足完整阶段、学习策略、原固定批次、带子步审计和至少 12/16 通过。完整验收包括前足初次支撑、小腿承重、后足补步、卸载无回碰、再次腾空、四足越墙及稳定落地 0.20 s。

```bash
cd /home/qyw/supermimic
export PYTHONPATH="/home/qyw/supermimic/third_party/supermimic_quadruped/src:/home/qyw/supermimic"
base=/home/qyw/supermimic/amp_supermimic/outputs/wall_jump_tracking
run="$base/calf-closed-loop-20261117-125605"
"$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.calf_tracking \
  --work "$run/balance-input" evaluate --stage full --strict-contact \
  --checkpoint "$run/full-balance/ckpt_latest.pt" --output "$run/full-balance-recheck" --export-amp
```


### 越墙后的落地同步

保留 `full-balance` 的 SuperMimic Actor，在四足完全越墙、机身开始下降之后，使用 `--touchdown-profile` 启用前足高度差反馈。它使用当前足端位置和关节雅可比调整大腿与小腿目标，修正量受配置上限约束。起跳与越墙阶段仍由原 Actor 执行。分阶段残差幅度和卸载后小腿间隙反馈在固定原批次上表现下降，作为实验记录保留。

完整策略使用原批次严格复验时通过 9/16，独立的 seed 1621 批次通过 11/16。同一反馈迁移到第二阶段 Actor 后，在原批次严格复验通过 7/16。结果低于 12/16 门槛，未导出 AMP。

```bash
cd /home/qyw/supermimic
export PYTHONPATH="/home/qyw/supermimic/third_party/supermimic_quadruped/src:/home/qyw/supermimic"
base=/home/qyw/supermimic/amp_supermimic/outputs/wall_jump_tracking
run="$base/calf-closed-loop-20261117-125605"
"$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.calf_tracking \
  --work "$run/balance-input" evaluate --stage full --strict-contact \
  --checkpoint "$run/full-balance/ckpt_latest.pt" \
  --touchdown-profile "$run/full-touchdown-selection.json" --output "$run/full-touchdown-recheck"
```


## 双后小腿接墙再下墙

用 `four_stage.py` 分别搜索后小腿接墙（`--phase catch`）和依次下墙（`--phase step`）。前三段的接触门槛包括真实第一次腾空、前腿支撑、后脚补步和前小腿卸载、真实第二次腾空、前足在墙后与两条后小腿在墙顶共同承重至少 0.12 秒。第四段要求后足分先后落到墙后，转移途中机身不再次腾空，最终四足稳定支撑至少 0.20 秒。墙侧、机身、大腿和落地过程的非预定碰撞用 200 Hz 接触数据检查。

本次名义起点的 3.40 秒轨迹由同一个 MuJoCo 自由机身仿真连续跑完。将同一组关节目标原样放到 seed 301 的 16 个扰动起点，第一跳和前小腿卸载 16/16，后小腿接墙 9/16，全段 0/16。该动作作为闭环训练的候选参考。

```bash
cd /home/qyw/supermimic
base=/home/qyw/supermimic/amp_supermimic/outputs/wall_jump_tracking
run="$base/four-stage-20261117-225449"
asset="$base/first-teacher-anchored-20261117-121208/input/wbl_tracking.xml"
"$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.four_stage_replay \
  --asset "$asset" --direct "$base/calf-brace-20261117-124549/calf-second.npz" \
  --candidate "$run/split-control.npz" --approach "$run/approach.npz" \
  --continuous "$run/continuous-control.npz" --output "$run/recheck"
```

比较两种方案的撞击时，使用同一支撑起点的 `direct` 和 `split`。力是各接触部件在竖直方向的力，冲量为第一次承重后 100 毫秒内的积分。后小腿对墙顶的载荷单列。


## 四段式 SuperMimic 闭环

`four_stage_tracking.py prepare` 从通过 MuJoCo 连续回放的四段式轨迹重新读取实际姿态和速度，生成 `descent`、`full` 两组跟踪片段。先训练 `descent`，再复用 Actor 和世界模型权重训练 `full`。两只后脚可以同时下墙。完整组要求真实第二次腾空、前双足在墙后和双后小腿在墙顶共同承重 0.12 秒，转移时机身不能再次腾空，最后四足稳定承重 0.20 秒。调参考的 `four_stage_refine.py` 使用不同 seed 的训练专用起点。

每次正式评估保存 `rollouts.npz` 和 `physics_200hz.npz`。后者包含 200 Hz 的接触力与 12 个关节的实际仿真驱动力矩，单位分别为 N 和 N·m。`evaluation.json` 汇总每个回合的各关节绝对峰值、触及 23.7 N·m 力矩上限附近的采样数，以及首次承重后 100 毫秒的双足合力峰值和冲量。

固定评估使用 seed 301 的原 16 个 XY 和朝向扰动。下墙评估从后小腿支撑态应用同样的扰动，完整评估从墙前地面启动。只有完整学习策略在原批次达到 12/16 才启用 AMP 导出。本次下墙最高 4/16，完整最高 2/16，均未导出。按接墙事件实时切换到单独下墙 Actor 的对照为 0/16。

```bash
cd /home/qyw/supermimic
export PYTHONPATH="/home/qyw/supermimic/third_party/supermimic_quadruped/src:/home/qyw/supermimic"
base=/home/qyw/supermimic/amp_supermimic/outputs/wall_jump_tracking
run="$base/four-stage-closed-loop-20261117-230312"
"$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.four_stage_tracking \
  --work "$run/transfer-input" evaluate --stage full \
  --checkpoint "$run/full-transfer/ckpt_0000200.pt" --output "$run/full-recheck"
```

## 靠近起跳、近膝接墙与前移小跳

`contact_route.py` 从上一版连续回放读入名义初始状态和关节目标作为搜索参考。模型使用原 WBL 资产和 30 cm × 8 cm 的墙。新轨迹从更靠近墙的位置起跳，要求前双小腿先承重，随后后脚弯曲补步。二跳前短暂卸下前小腿，再验证真实腾空、前足落地、双后小腿近膝接墙、机头下俯和前足承重。最后允许向前迈步或一次短暂腾空，要求越过墙后四足稳定支撑 0.20 秒。名义回放使用了诊断选项 `--max-hops 3`，记录了 3 次短暂失去支撑。单次小跳的搜索和两批扰动验收尚未通过。阶段搜索固定前一段控制序列，每次从同一墙前状态连续仿真。

`evaluation.json` 记录名义起点和两批各 16 个原扰动起点的结果。它还记录机身俯仰、膝部接触位置、前足承重比例、200 Hz 接触峰值与关节力矩。`replay.npz` 保存仿真实际状态和关节目标。该目录是 MuJoCo 参考实验，扰动结果用于决定是否值得做后续闭环跟踪和 AMP 导出。

复验已保存的分段解并重渲染视频。

```bash
cd /home/qyw/supermimic
base=amp_supermimic/outputs/wall_jump_tracking
run="$base/contact-route-v2"
asset="$base/first-teacher-anchored-20261117-121208/input/wbl_tracking.xml"
"$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.contact_route \
  --asset "$asset" --baseline "$base/four-stage-20261117-225449/continuous-replay.npz" \
  --output "$run" --resume --max-hops 3
MUJOCO_GL=egl "$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.contact_route_view \
  --asset "$asset" --run "$run"
```

### 第一跳由前腿主推

`--front-led` 从 200 Hz 实际接触力计算开始到第一次离地的竖直冲量，要求前脚承担至少 65%。后膝在起跳时允许随姿态伸展。搜索约束起跳期间膝弯曲 88°–138°，并要求搭墙前仍有至少 83° 弯曲。`--first-only` 在前双小腿建立 0.10 秒稳定支撑后结束，可用于单独复验第一跳。

```bash
cd /home/qyw/supermimic
base=amp_supermimic/outputs/wall_jump_tracking
run="$base/front-led-takeoff"
asset="$base/first-teacher-anchored-20261117-121208/input/wbl_tracking.xml"
"$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.contact_route \
  --asset "$asset" --baseline "$base/four-stage-20261117-225449/continuous-replay.npz" \
  --output "$run" --first-only --front-led --resume
MUJOCO_GL=egl "$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.contact_route_view \
  --asset "$asset" --run "$run"
```

### 新版第一跳接完整四段

`front-led-four-stage` 复用 `front-led-takeoff` 的前 26 个关节目标，在同一场 MuJoCo 仿真中接上补步、二跳和下墙。补步验收要求后双足相对搭墙时至少各前移 3 cm，并恢复屈膝承重。扰动使用 seed 301 和 1621 的原批次。在各控制段切换前，检查该段是否通过并累计到后续通过率。二跳诊断另记真实腾空及前足与后小腿同时支撑的次数。

```bash
cd /home/qyw/supermimic
base=amp_supermimic/outputs/wall_jump_tracking
run="$base/front-led-four-stage"
asset="$base/first-teacher-anchored-20261117-121208/input/wbl_tracking.xml"
"$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.contact_route \
  --asset "$asset" --baseline "$base/four-stage-20261117-225449/continuous-replay.npz" \
  --output "$run" --front-led --resume
MUJOCO_GL=egl "$base/.venv/bin/python" -B -m amp_supermimic.wall_jump_tracking.contact_route_view \
  --asset "$asset" --run "$run"
```
