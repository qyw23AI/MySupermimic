# ROBOCON 2027 障碍赛专家数据审计与补充路线

审计日期：2026-11-16。依据用户提供的规则V2.0第3.1节表1、第4.3节表2；不是对未来规则修订的保证。

## 结论与实测
当前27份非淘汰分组报告：walk4、turn7、jump7、crawl2、run6、strafe1。22份通过旧运动学门槛，只有turn/ai4_dog_synth_eight_walk通过旧平地PD门槛。该通过不是比赛合格证。旧判据有局限：要求支撑帧占比>0.5、前进比例>0.5，不适合所有跳跃和原地转向；其失败不能全部归结为动作无效。反过来，IK通过也不意味着可执行。

已有jump包括4种翻转、1m前跳、NJU jump_1、kine2go_rollout_jump_b。翻转不是本比赛优先技能。jump_1 的足端P95误差约9.1cm、关节速度P99约26.9rad/s，未过旧运动学门槛。walk内两条马动作有回放跌倒记录。不要仅因train.json存在就全量采样。

机器可读逐条结果：data/obstacle_audit.json。未读取rubbish内动作，也未恢复任何淘汰动作。当前障碍专家白名单为空，避免把“候选”误报为“比赛专家”。

## 障碍—数据映射
|障碍与规则|现有可借鉴内容|缺少的关键专家|
|---|---|---|
|绕桩：直径10cm，中心距90cm，S形经过必达区|保留walk/turn；eight轨迹优先做种子|小半径连续转向、停止再起步、路径/全身投影约束|
|限高：内高29cm，宽长80cm|crawl可作慢速支撑节律先验|整机含传感器最高点约束、降低躯干并连续行走；爬行名称不代表低姿态|
|高墙：高30cm、厚8cm、宽80cm|前跳参考的收腿/落地节律|整机和足端越墙净空、起跳距离、冲量、落地恢复；不是30cm高平台攀爬|
|沟壑组合：12cm高/32cm深台阶、36cm高平台、36cm沟宽、20°坡|walk及前跳只能初始化|上下每级至少一足接触；平台起跳/落地；上下坡和平台转向|
|砖木坑：15cm护栏，碎砖和10–20mm树皮|慢走/小跑先验|跨边缘、随机支撑面、滑动与扰动恢复|
|轮胎：165/70R13，2×4阵|慢速多足支撑先验|避开胎孔或可靠跨越、曲面接触，不能把轮胎简化成平板后宣称通过|
|梅花桩：20×20×10cm，每个顶面至少踩一次|慢爬步态仅作初始化|落脚位置、足缘余量、按序覆盖全部桩面，不能大跳略过|
|桥：长3m、宽40cm、高10cm|直行慢走|收窄足迹、上桥/下桥与侧向恢复|

场地尺寸允许±5%。把这个范围加入验证集；限高最小高度27.55cm、高墙最大31.5cm、沟最大37.8cm，另设机器人几何/感知余量。余量不能简单加到所有尺寸：窄桥和桩面应按最小宽度验证。

## 本轮实际新增与重定向
来源：https://huggingface.co/datasets/MIMUW-Robotics/kine2go/tree/244dc58/data/solo8_jump_forward_b/traj_0001
新增1条既有跳跃策略的不同rollout，不是新增独立技能。571帧、60Hz；官方卡说明base_quat为WXYZ，转换XY​​ZW；关节顺序取该clip的config.env_cfg.dof_names，不能按其他格式硬套。原始文件用torch.load(weights_only=True)读取，保留SHA256与许可。

- 原始：data/raw/robocon_jump_candidates/
- Go2转换：data/go2/robocon_jump_candidates/
- WBL：data/wblwbl/obstacle_candidates/{baseline,neutral_ground}/，各476帧、50Hz。
- baseline：旋转初始足端校准，足端P95约0.124mm，前进比例-0.0969，关节跟踪RMSE0.270rad。
- neutral_ground：中立站姿+地面对齐，足端P95约1.50mm，前进比例-0.0125，关节跟踪RMSE0.302rad。

两版均未通过物理回放；保留为隔离候选，train.json改为candidate_manifest.json，不接入walk/turn训练。拟合误差只衡量目标足端拟合，不代表物理运动质量。现有回放使用固定Kp40/Kd4和模型读取的力矩限值，还需与实机电机、控制周期、实际HIM动作接口核对。未完成障碍场景回放，不声称跳过36cm沟壑。

复现：
```bash
/home/qyw/miniconda3/envs/himloco/bin/python -B amp_supermimic/scripts/prepare_obstacle_jump.py
```
依赖trackerlab_amp环境、现有WBL MJCF及trackerlab子模块。输出已存在时跳过，不覆盖人工筛选结果。

## 外部来源筛选
1. Kine2Go：https://nomagiclab.github.io/kine2go-pipeline/ 与上述Hugging Face数据卡。800条是40种策略的rollout，不能当800种技能；适合增加初始姿态/速度变化，但没有因此获得比赛台阶攀爬。
2. Extreme Parkour：https://github.com/chengxuxin/extreme-parkour 。价值是训练环境、课程和教师策略路线，不是可直接丢给WBL的专家CSV。适合建立高台、沟壑、窄落脚教师，然后采集WBL自身成功轨迹。
3. SoloParkour：https://gepetto.github.io/SoloParkour/ 。借鉴爬高台、跨沟、低姿态的约束RL任务定义。Solo形态与WBL不同，需重训而非复制关节角。
4. Imitation_from_video：https://github.com/arclab-hku/Imitation_from_video 。检查当前main文件树，实际公开video_motion目录主要为limp/walk+turn，README后空翻演示不等于公开了对应越墙专家。未为凑数量下载不匹配数据。
5. Quad-Imaginarium：https://github.com/Amap-Robotics/Quad-Imaginarium 。当前README仍写full dataset将在论文接收后发布，不把7488条计入可用库存。

## 如何真正提高质量
优先直接为WBL生产任务专家，而不是继续无限扩充动物运动学轨迹。

A. 先建逐障碍地形与规则检查器。成功必须同时满足前进、足端接触覆盖、无越界/穿透、落地稳定；台阶/梅花桩用contact geom ID记录每个面是否实际踩过。记录障碍坐标、尺寸、阶段、足端接触、root/q/dq、执行动作、力矩、传感器历史、done/reset前状态。

B. 易到难WBL教师训练：平地转弯→5/8/10/12cm台阶→10/15/20°坡→20/28/36cm沟→10/20/30cm墙。教师可用特权高度图、精确脚点/障碍位置、PPO或轨迹优化；它的目的只是生成物理有效专家，不要求教师本身已经使用AMP或SuperMimic。上下楼、起跳和落地分别有阶段目标，奖励不能仅保留平地速度追踪。

C. 跳跃重定向升级：联合优化root和关节轨迹，约束起跳/腾空/落地接触、非穿透、摩擦锥、关节/速度/力矩、目标落脚区。时间缩放后重新计算速度和接触动力学；不得简单把1m跳跃横向压缩成36cm。30cm墙需全身碰撞几何净空检查，36cm沟需在36cm高平台场景中验证。

D. 采集与筛选：每个任务至少3训练种子，先收集每尺寸档20条成功完整episode作为起步目标（数量不是质量保证）。保留进场—动作—恢复—出场，不裁掉失败落地。用独立种子和±5%尺寸验证；候选至少重复10次、成功≥9次才进入初步白名单，随后加摩擦/质量/电机/延迟扰动再评估。记录失败原因，不只记录总reward。

E. AMP按技能/地形条件采样，分开walk、turn、crawl、stairs、gap、wall、precision_step。保留平地先验但允许越障姿态偏离；未经条件化混合翻滚和慢走会使奖励目标冲突。当前24维q/dq特征看不见root运动、足端相对地形、障碍位置，必须扩展或配合明确任务奖励。HIE本体历史也不会自动提前看见沟壑，需要高度图/深度或可靠的任务几何输入。

F. 世界模型只使用带真实执行动作的WBL仿真转移；新下载Go2的actions不能直接当WBL动作标签，IK CSV也不能直接训练WBL动力学。模型想象和可微AMP更新待真实任务基线通过后接入。

## 尚未完成
已完成规则分析、27条报告审计、新数据获取、两版WBL重定向与平地PD回放。尚无新增通过验证的障碍专家，尚未实现WBL障碍教师训练/接触约束优化、比赛地形回放和专家视频；不得宣称数据缺口已补齐。下一优先级是12cm上下台阶教师和36cm沟壑教师，它们比继续下载翻转动作更直接对应比赛得分。
