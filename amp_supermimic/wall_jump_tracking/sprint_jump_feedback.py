"""GPU search and independent MuJoCo replay of contact-triggered WBL wall jumps."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from .brace_experiment import FEET, HOME, SUBSTEPS, make_model
from .four_stage import force_signals
from .contact_route import add_sensors
from .single_hop import perturb
from .sprint_jump import wall_assist


class Controller:
    def __init__(self, model, runup, params, device):
        self.params = torch.as_tensor(params, dtype=torch.float32, device=device)
        self.phase = torch.zeros(len(params), dtype=torch.long, device=device)
        self.home = torch.tensor(HOME, dtype=torch.float32, device=device)
        self.runup = torch.as_tensor(runup, device=device)
        self.bounds = torch.as_tensor(model.jnt_range[1:], dtype=torch.float32, device=device)
        self.neutral = self.home.expand(len(params), -1).clone()
        self.height = torch.zeros(len(params), device=device)
        self.floor = [model.sensor('floor_' + foot).adr[0] for foot in FEET]
        self.floor_force = [a + 1 for a in self.floor]
        self.rear_toe = [model.sensor('position_' + foot).adr[0] for foot in FEET[:2]]
        self.wall = [model.sensor('wall_' + foot).adr[0] for foot in FEET[:2]]
        self.front_x = [model.sensor('position_' + foot).adr[0] for foot in FEET[2:]]
        self.rear_knees = [model.sensor('knee_' + leg).adr[0] for leg in ('RR', 'RL')]
        self.contact_x = self.height.clone()
        self.braking = torch.zeros_like(self.phase,dtype=torch.bool)
        self.refined = self.params.shape[1] >= 10
        self.recover = self.params.shape[1] >= 18
        self.gait_phase = self.height.clone()
        self.support_frames = self.height.clone()

    def action(self, q, v, sensors, tick):
        distance, depth, bend, rear_sweep, front_sweep, drive, tuck = self.params[:, :7].T
        loaded = sensors[:, self.floor].gt(.5)
        on_wall = sensors[:, self.wall].gt(.5).any(1)
        start = (self.phase == 0) & (q[:,0] > -distance) & loaded[:, :2].any(1) & loaded[:, 2:].any(1) & (v[:,0] > .55)
        if self.params.shape[1] >= 52:
            start = (self.phase == 0) & loaded.all(1) & (v[:,:3].norm(dim=1)<.12) & (tick>=31)
        self.neutral[start] = q[start, 7:]
        self.height[start] = q[start, 2]
        phase = self.phase.clone()
        self.phase[start] = 1
        compression = self.height - q[:,2]
        push = (phase == 1) & loaded[:, :2].any(1) & (compression >= depth) & (v[:,2] > -.8)
        if self.params.shape[1] >= 32:
            rear_load = sensors[:,self.floor_force[:2]].abs()
            weaker_share = rear_load.min(1).values / rear_load.sum(1).clamp_min(1.)
            push &= weaker_share >= self.params[:,29]
        if self.params.shape[1] in (39,40):
            trigger_lead = self.params[:,36]
            early = ((phase == 1) & loaded[:, :2].any(1)
                     & (compression >= (depth-trigger_lead).clamp_min(.008))
                     & (v[:,2] > -.8) & (v[:,0] > .65))
            share = self.params[:,39] if self.params.shape[1] == 40 else self.params[:,29]
            push |= early & (weaker_share >= share)
        if self.params.shape[1] in (41,47,52,56):
            speed_limit, front_sweep_fix, rear_sweep_fix, share_limit, min_depth = self.params[:,36:41].T
            self.braking |= (phase == 1) & (v[:,0] < speed_limit)
            push |= ((phase == 1) & self.braking & (compression >= min_depth)
                     & (weaker_share >= share_limit) & loaded[:,:2].all(1) & (v[:,2] > -.8))
        if self.params.shape[1] >= 52:
            push = ((phase == 1) & loaded[:,:2].all(1)
                    & (compression >= self.params[:,48]) & (v[:,2] > -.6))
        self.phase[push] = 2
        airborne = (phase == 2) & ~loaded.any(1) & (v[:,2] > .10)
        self.phase[airborne] = 3
        if self.refined:
            front_fold, rear_fold, clearance = self.params[:, 7:10].T
            front_over = (sensors[:, self.front_x] > .064).all(1)
            knees_over = (sensors[:, self.rear_knees] > clearance[:, None]).all(1)
            knees_high = (sensors[:, [a+2 for a in self.rear_knees]] > .325).all(1)
            toe_clear = True
            if self.params.shape[1] >= 25:
                toe_clear = (sensors[:, [a+2 for a in self.rear_toe]] > .325).all(1)
            if self.params.shape[1] >= 27:
                # Keep the right foot tucked until its sphere clears the near lip.
                toe_clear = toe_clear & (sensors[:, self.rear_toe[0]] > self.params[:,26])
            self.phase[(phase == 3) & front_over & knees_over & knees_high & toe_clear & (v[:,2] < .35)] = 4
            touch = (phase == 4) & on_wall
            self.contact_x[touch] = q[touch,0]
            self.phase[touch] = 6
            self.phase[(phase == 6) & (~on_wall | (q[:,0] - self.contact_x > .045))] = 7
            self.phase[(phase == 7) & loaded[:,2:].all(1) & (q[:,0] > .13)] = 5
        else:
            self.phase[(phase == 3) & on_wall & (sensors[:, self.front_x] > .04).all(1)] = 4
        self.phase[((phase == 3) | (phase == 4)) & (q[:,0] > .13) & loaded[:, 2:].any(1) & ~on_wall] = 5
        target = self.runup[min(tick, len(self.runup)-1)].expand_as(self.neutral).clone()
        crouch = self.neutral.clone()
        crouch[:, 1::3] += .30
        crouch[:, 2::3] -= bend[:, None]
        if self.params.shape[1] >= 34:
            rear, front = self.params[:,32:34].T
            crouch[:,[2,5]] += (bend - rear)[:,None]
            crouch[:,[8,11]] += (bend - front)[:,None]
            crouch[:,[7,10]] -= .22
        if self.params.shape[1] >= 32:
            # Mirror the measured crouch pose across the sagittal plane.
            legs = crouch.reshape(-1,4,3)
            sign = q.new_tensor([-1.,1.,1.])
            mirror = legs[:,[1,0,3,2]] * sign
            crouch = torch.lerp(legs, mirror, .5*self.params[:,30,None,None]).flatten(1)
        if self.params.shape[1] == 56:
            rear_pose, front_pose, rear_drive, front_drive = self.params[:,52:56].T
            crouch[:,[1,4]] += rear_pose[:,None]
            crouch[:,[7,10]] += front_pose[:,None]
        extend = crouch.clone()
        if self.params.shape[1] in (41,47,52,56):
            crouch[:,[7,10]] += (front_sweep_fix*self.braking)[:,None]
            extend[:,[1,4]] += (rear_sweep_fix*self.braking)[:,None]
        extend[:, [1,4]] += rear_sweep[:, None]
        extend[:, [7,10]] += front_sweep[:, None]
        extend[:, 2::3] += drive[:, None]
        if self.params.shape[1] >= 32:
            split = self.params[:,31,None]
            extend[:,[2,5]] += split
            extend[:,[8,11]] -= split
        if self.params.shape[1] in (39,40):
            front_extension, rear_extension = self.params[:,37:39].T
            extend[:,[8,11]] += front_extension[:,None]
            extend[:,[2,5]] += rear_extension[:,None]
        if self.params.shape[1] >= 52:
            rear, front, drive_delta = self.params[:,49:52].T
            extend[:,[1,4]] += rear[:,None]
            extend[:,[7,10]] += front[:,None]
            extend[:,2::3] += drive_delta[:,None]
            if self.params.shape[1] == 56:
                extend[:,[2,5]] += rear_drive[:,None]
                extend[:,[8,11]] += front_drive[:,None]
        flight = self.home.expand_as(target).clone()
        flight[:, 1::3] += .65
        flight[:, 2::3] -= tuck[:, None]
        if self.refined:
            flight[:, [8,11]] -= front_fold[:, None]
            flight[:, [2,5]] -= rear_fold[:, None]
            flight[:, [1,4]] += .25
        assist = flight.clone()
        assist[:, [2,5]] += drive[:, None]
        assist[:, [1,4]] += rear_sweep[:, None]
        target = torch.where((self.phase == 1)[:,None], crouch, target)
        target = torch.where((self.phase == 2)[:,None], extend, target)
        target = torch.where((self.phase == 3)[:,None], flight, target)
        target = torch.where((self.phase == 4)[:,None], assist, target)
        if self.refined:
            target = torch.where((self.phase == 6)[:,None], assist, target)
            release = flight.clone()
            release[:, 6:] = self.home[6:]
            release[:, :6] = torch.where((q[:,0] > .37)[:,None], self.home[:6], release[:, :6])
            target = torch.where((self.phase == 7)[:,None], release, target)
        target = torch.where((self.phase == 5)[:,None], self.home, target)
        if self.recover:
            target = self.recovery_target(q, v, sensors, loaded, target)
        if self.params.shape[1] >= 21:
            target = self.balance_target(q, v, sensors, target)
        if self.params.shape[1] >= 47:
            target = self.landing_sequence(q,v,sensors,loaded,target)
        return target.clamp(self.bounds[:,0], self.bounds[:,1])


    def landing_sequence(self,q,v,sensors,loaded,target):
        release_x, rear_fold, rear_sweep, front_buffer, rear_support, pitch_trim = self.params[:,41:47].T
        # Keep each rear foot folded independently until its sphere is beyond the wall.
        rear_x = sensors[:,self.rear_toe]
        settling = (self.phase == 5) | (self.phase == 7) | (self.phase == 8)
        near_edge = settling[:,None] & (rear_x < release_x[:,None]) & ~loaded[:,:2]
        target[:,[2,5]] -= rear_fold[:,None] * near_edge
        target[:,[1,4]] += rear_sweep[:,None] * near_edge
        # Shape the first front landing by vertical speed, then restore the stance.
        plunge = (self.phase == 5) & (v[:,2] < -.18)
        impact = (-v[:,2]).clamp(0.,1.5) * plunge
        target[:,[8,11]] -= front_buffer[:,None] * impact[:,None]
        target[:,[7,10]] += pitch_trim[:,None] * impact[:,None]
        # Lift the rear knee away from the floor when the rear foot is already loaded.
        knees_z = sensors[:,[a+2 for a in self.rear_knees]]
        low_knee = ((.15-knees_z)/.10).clamp(0.,1.)
        target[:,[2,5]] += rear_support[:,None] * low_knee * settling[:,None] * loaded[:,:2] * ~near_edge
        return target

    def balance_target(self, q, v, sensors, target):
        hip, knee, damping = self.params[:,18:21].T
        w, x, y, z = q[:,3:7].T
        roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        active = (self.phase >= 2) & (self.phase != 8)
        correction = (roll + damping*v[:,3]).clamp(-.7,.7) * active
        if self.params.shape[1] >= 22:
            grounded = (self.phase == 1) | (self.phase == 2)
            flying = self.phase == 3
            correction = (roll + damping*v[:,3]).clamp(-.6,.6) * (
                grounded + flying * self.params[:,21])
        # Common hip offset and opposite knee offsets act on different axes.
        target[:,0::3] += (hip*correction).clamp(-.35,.35)[:,None]
        target[:,2::3] += (knee*correction).clamp(-.42,.42)[:,None] * q.new_tensor([1.,-1.,1.,-1.])
        if self.params.shape[1] >= 25:
            load_gain, lateral_gain, toe_fold = self.params[:,22:25].T
            support = sensors[:,self.floor_force].abs()
            imbalance = (support[:,::2].sum(1) - support[:,1::2].sum(1)) / support.sum(1).clamp_min(25.)
            grounded = (self.phase == 1) | (self.phase == 2)
            drive = (load_gain * imbalance).clamp(-.26,.26) * grounded
            target[:,2::3] += drive[:,None] * q.new_tensor([1.,-1.,1.,-1.])
            target[:,0::3] += (lateral_gain*v[:,1]).clamp(-.22,.22)[:,None] * grounded[:,None]
            above = sensors[:,self.rear_toe[0]+2] > .327
            target[:,2] -= toe_fold * (self.phase == 3) * ~above
            if self.params.shape[1] >= 27:
                near_wall = (sensors[:,self.rear_toe[0]] > -.32) & (sensors[:,self.rear_toe[0]] < .06)
                target[:,1] += self.params[:,25] * (self.phase == 3) * near_wall
        if self.params.shape[1] >= 29:
            launch_gain, flight_gain = self.params[:,27:29].T
            grounded = (self.phase == 1) | (self.phase == 2)
            flying = self.phase == 3
            trim = (launch_gain*(roll + .16*v[:,3])*grounded
                    + flight_gain*(roll + .08*v[:,3])*flying).clamp(-.30,.30)
            target[:,0::3] += trim[:,None]
            target[:,2::3] += .65*trim[:,None] * q.new_tensor([1.,-1.,1.,-1.])
        if self.params.shape[1] >= 36:
            rear_trim, front_trim = self.params[:,34:36].T
            # Apply lateral corrections only while the corresponding pair can push.
            push = self.phase == 2
            loaded = sensors[:,self.floor].gt(.5)
            correction = (v[:,1] - .025).clamp(-.32,.32) * push
            target[:,[0,3]] += (rear_trim*correction)[:,None] * loaded[:,:2]
            target[:,[6,9]] += (front_trim*correction)[:,None] * loaded[:,2:]
        return target

    def recovery_target(self, q, v, sensors, loaded, target):
        bend, stride, lift, frequency, brake, pitch_gain, stop_speed, min_x = self.params[:,10:18].T
        walking = self.phase == 5
        feet_over = (sensors[:, self.floor] > .5).sum(1) >= 2
        speed = torch.linalg.vector_norm(v[:,:3], dim=1)
        supported = walking & feet_over & (speed < stop_speed) & (q[:,0] > min_x)
        self.support_frames = torch.where(supported, self.support_frames + 1, 0.)
        self.phase[walking & (self.support_frames >= 3)] = 8
        walking = self.phase == 5
        self.gait_phase += walking * frequency * (.02 * 2 * torch.pi)
        envelope = (v[:,0].abs() / .85).clamp(.12, 1.)
        swing = torch.sin(self.gait_phase[:,None] + q.new_tensor([0.,torch.pi,torch.pi,0.]))
        vertical = (-v[:,2]).clamp(0.,1.6)
        compress = bend * vertical
        pitch = 2 * (q[:,3]*q[:,5] - q[:,6]*q[:,4])
        pitch_offset = pitch_gain * (pitch + .12*v[:,4]).clamp(-.7,.7)
        gait = self.home.expand_as(target).clone()
        gait[:,1::3] += .5*compress[:,None] + stride[:,None]*envelope[:,None]*swing
        gait[:,1::3] -= (brake * v[:,0].clamp(0.,2.))[:,None] * loaded
        gait[:,1::3] += pitch_offset[:,None] * q.new_tensor([-1.,-1.,1.,1.])
        gait[:,2::3] -= compress[:,None] + lift[:,None]*envelope[:,None]*swing.clamp_min(0.)
        target = torch.where((self.phase == 8)[:,None], self.home, target)
        return torch.where(walking[:,None], gait, target)


def replay(model, starts, params, runup, frames, gpu, *, keep_on_device=False):
    if gpu:
        import mujoco_warp as mw
        import warp as wp

        wp.init()
        with wp.ScopedDevice('cuda:0'):
            seed = mj.MjData(model)
            mj.mj_forward(model, seed)
            m = mw.put_model(model)
            d = mw.put_data(model, seed, nworld=len(starts), nconmax=64, njmax=256)
            q, v, sens, ctrl = map(wp.to_torch, (d.qpos, d.qvel, d.sensordata, d.ctrl))
            q.copy_(torch.as_tensor(starts[:,1:20], device=q.device))
            v.copy_(torch.as_tensor(starts[:,20:38], device=q.device))
            mw.forward(m, d)
            controller = Controller(model, runup, params, 'cuda')
            history = q.new_empty((frames*SUBSTEPS,len(starts),38))
            observed = sens.new_empty((frames*SUBSTEPS,len(starts),model.nsensordata))
            for tick in range(frames):
                ctrl.copy_(controller.action(q,v,sens,tick))
                for substep in range(SUBSTEPS):
                    mw.step(m,d)
                    k = tick*SUBSTEPS+substep
                    history[k,:,0] = 0
                    history[k,:,1:20] = q
                    history[k,:,20:] = v
                    observed[k].copy_(sens)
            wp.synchronize()
            if keep_on_device:
                return history.transpose(0,1), observed.transpose(0,1)
            return history.transpose(0,1).cpu().numpy(), observed.transpose(0,1).cpu().numpy()
    sim = [mj.MjData(model) for _ in starts]
    for data, start in zip(sim, starts):
        data.qpos[:] = start[1:20]
        data.qvel[:] = start[20:38]
        mj.mj_forward(model,data)
    controller = Controller(model, runup, params, 'cpu')
    raw = np.empty((len(sim),frames*SUBSTEPS,38))
    sensors = np.empty((len(sim),frames*SUBSTEPS,model.nsensordata))
    for tick in range(frames):
        q,v,sens = [torch.tensor(np.stack([getattr(d,k) for d in sim]), dtype=torch.float32)
                    for k in ('qpos','qvel','sensordata')]
        action = controller.action(q,v,sens,tick).numpy()
        for i,data in enumerate(sim):
            data.ctrl[:] = action[i]
            for substep in range(SUBSTEPS):
                mj.mj_step(model,data)
                k = tick*SUBSTEPS+substep
                raw[i,k] = np.r_[data.time,data.qpos,data.qvel]
                sensors[i,k] = data.sensordata
    return raw,sensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--recovery', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--robust', action='store_true')
    p.add_argument('--standing-forward', action='store_true')
    p.add_argument('--standing-jump', action='store_true')
    p.add_argument('--landing-sequence', action='store_true')
    p.add_argument('--braking-recovery', action='store_true')
    p.add_argument('--calibrate-crashes', action='store_true')
    p.add_argument('--launch-timing', action='store_true')
    p.add_argument('--push-lateral', action='store_true')
    p.add_argument('--micro-lateral', action='store_true')
    p.add_argument('--protect-clearance', action='store_true')
    p.add_argument('--stabilize-shallow', action='store_true')
    p.add_argument('--shallow-crouch', action='store_true')
    p.add_argument('--clean-symmetric', action='store_true')
    p.add_argument('--shape-push', action='store_true')
    p.add_argument('--roll-launch', action='store_true')
    p.add_argument('--toe-clearance', action='store_true')
    p.add_argument('--launch-balance', action='store_true')
    p.add_argument('--balance-early', action='store_true')
    p.add_argument('--balance', action='store_true')
    p.add_argument('--retain-stance', action='store_true')
    p.add_argument('--holdout', type=int, action='append', default=[])
    p.add_argument('--warm', type=Path)
    p.add_argument('--clear-rear', action='store_true')
    args = p.parse_args()
    if args.robust and (not args.recovery or args.batch % 4):
        p.error('--robust requires --recovery and a batch divisible by four')
    args.output.mkdir(parents=True, exist_ok=True)
    model = make_model(args.asset, configure=add_sensors)
    # Warp's multiccd path does not support margins on convex pairs.
    model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_MULTICCD)
    with np.load(args.baseline) as baseline:
        initial, runup, prior = (baseline[k].copy() for k in ('initial','runup','params'))
    rng = np.random.default_rng(20261119)
    low = np.array([.40,.025,.40,-.65,-.55,.7,.55,0.,.05,-.035])
    high = np.array([.86,.115,1.15,.65,.75,2.0,1.4,.65,.65,.12])
    best = -np.inf
    center = np.r_[prior, .35, .35, .035]
    frames = 205 if args.recovery else 135
    if args.recovery:
        low = np.r_[prior, .05, -.36, .10, 1.6, -.25, -.7, .20, .28]
        high = np.r_[prior, .65, .42, .85, 4.0, .38, .8, .38, .80]
        center = (low + high) / 2
    if args.clear_rear:
        if not (args.robust and (len(prior) == 18 or args.launch_balance and len(prior) == 22)):
            p.error('--clear-rear needs an 18-parameter recovery baseline and --robust')
        center = prior.copy()
        if args.warm:
            with np.load(args.warm) as warm:
                center = warm['params'].copy()
        spread = np.array([.035,.035,.22,.22,.10,.23,0.,0.,0.,0.,
                           .20,.24,.27,.75,.18,.32,.08,.18])
        if args.retain_stance:
            spread *= .48
        if args.launch_balance:
            spread = np.r_[spread,np.zeros(4)]
        low, high = prior - spread, prior + spread
        low[1] = max(low[1], .015)
        low[2] = max(low[2], .10)
    if args.balance:
        if not args.clear_rear:
            p.error('--balance requires --clear-rear')
        # Keep the existing maneuver parameters and search only feedback gains.
        low = np.r_[prior, -.80, -.90, .025]
        high = np.r_[prior, .80, .90, .24]
        center = np.r_[prior, 0.,0.,.12]
        if args.balance_early:
            low, high, center = np.r_[low,0.], np.r_[high,.35], np.r_[center,.10]
    if args.launch_balance:
        low = np.r_[prior,-.60,-.90,0.]
        high = np.r_[prior,.60,.90,.55]
        center = (low + high)/2
        if args.toe_clearance:
            low, high = np.r_[low,-.5,-.055], np.r_[high,.5,.025]
            center = np.r_[center,0.,-.025]
            if args.warm:
                with np.load(args.warm) as warm:
                    saved = warm['params']
                    center = saved.copy() if len(saved) == 27 else np.r_[saved,0.,-.025]
        if args.roll_launch:
            low, high, center = np.r_[low,-1.2,-.38], np.r_[high,1.2,.38], np.r_[center,0.,0.]
    if args.shape_push:
        if not (args.roll_launch and args.warm):
            p.error('--shape-push requires --roll-launch and a warm profile')
        with np.load(args.warm) as warm:
            center = np.r_[warm['params'],.08,.45,0.]
        low = np.r_[low,0.,0.,-.55]
        high = np.r_[high,.27,1.,.55]
    if args.clean_symmetric:
        if not args.shape_push:
            p.error('--clean-symmetric requires --shape-push')
        with np.load(args.warm) as warm:
            center = warm['params'].copy()
        spread = np.zeros_like(center)
        indices = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,25,26,31]
        spread[indices] = [.04,.018,.16,.23,.28,.16,.15,.13,.045,.17,.22,.24,.55,.16,
                           .22,.08,.14,.18,.035,.28]
        low, high = center - spread, center + spread
    if args.shallow_crouch and not args.stabilize_shallow:
        if not args.clean_symmetric:
            p.error('--shallow-crouch requires --clean-symmetric')
        center = np.r_[center,.45,.13]
        low, high = np.r_[low,.22,.02], np.r_[high,.70,.23]
        low[1], high[1], center[1] = .015, .065, .035
    if args.stabilize_shallow:
        if not (args.shallow_crouch and args.clean_symmetric and (len(center) == 34 or args.launch_timing and len(center) == 36)):
            p.error('--stabilize-shallow requires a warm 34-parameter shallow-crouch profile')
        prior = center.copy()
        spread = np.zeros_like(center)
        indices = [10,11,12,13,14,15,16,17,22,23,27,28]
        spread[indices] = [.16,.17,.18,.50,.16,.23,.07,.13,.065,.22,.13,.11]
        low, high = center - spread, center + spread
        low[16] = max(low[16], .16)
        if args.protect_clearance:
            spread[22:29] *= .22
            low, high = center - spread, center + spread
            low[16] = max(low[16], .16)
    if args.micro_lateral:
        spread = np.zeros_like(center)
        spread[[10,11,12,13,14,15,16,17,23]] = [.055,.06,.065,.22,.05,.08,.028,.045,.10]
        low, high = center - spread, center + spread
        low[16] = max(low[16], .16)
    if args.push_lateral:
        if not args.stabilize_shallow:
            p.error('--push-lateral requires --stabilize-shallow')
        center = np.r_[prior,0.,0.]
        low, high = np.r_[prior,-1.2,-1.2], np.r_[prior,1.2,1.2]
    if args.launch_timing:
        center = np.r_[prior,0.,0.,0.]
        low, high = np.r_[prior,0.,-.40,-.40], np.r_[prior,.032,.40,.40]
    if args.calibrate_crashes:
        center = np.r_[center,.13]
        low, high = np.r_[low,.02], np.r_[high,.27]
        high[36] = .055
    if args.braking_recovery:
        if not args.calibrate_crashes:
            p.error('--braking-recovery requires --calibrate-crashes')
        low = np.r_[prior,.28,-.40,-.68,.04,.012]
        high = np.r_[prior,.85,.40,.68,.27,.045]
        center = (low+high)/2
    if args.landing_sequence:
        if not (args.robust and args.recovery and len(prior)==41):
            p.error('--landing-sequence requires a 41-parameter baseline and --robust --recovery')
        low = np.r_[prior,.045,0.,-.55,-.25,0.,-.30]
        high = np.r_[prior,.155,.80,.32,.42,.55,.35]
        center = np.r_[prior,.066,0.,0.,0.,0.,0.]
    if args.standing_jump:
        if not (args.robust and args.recovery and len(prior)==47):
            p.error('--standing-jump requires a 47-parameter baseline and --robust --recovery')
        low = np.r_[prior,.32,.012,-.65,-.55,-.65]
        high = np.r_[prior,.69,.085,.70,.75,.50]
        center = (low+high)/2
        runup = np.repeat(np.asarray(HOME,dtype=np.float32)[None],205,axis=0)
    if args.standing_forward:
        if not args.standing_jump:
            p.error('--standing-forward requires --standing-jump')
        low[47:] = [.32,.012,-1.25,-1.25,-1.35]
        high[47:] = [.70,.095,1.25,1.25,.65]
        low, high = np.r_[low,-.50,-.42,-.95,-.95], np.r_[high,.50,.55,.95,.95]
        center = (low+high)/2
    for iteration in range(0 if args.eval_only else args.rounds):
        count = args.batch // 4 if args.robust else args.batch
        population = rng.uniform(low,high,(count,len(low)))
        population[count//3:] = np.clip(
                rng.normal(center,.18*(high-low),(count-count//3,len(low))),low,high)
        population[0] = center
        starts = np.repeat(initial[None],args.batch,axis=0)
        if args.robust:
            training_starts = np.vstack([initial, perturb(initial, 8103, 3)])
            if args.calibrate_crashes:
                training_starts[2] = perturb(initial,301,16)[1]
                training_starts[3] = perturb(initial,1621,16)[10]
            if args.landing_sequence:
                training_starts = np.vstack([initial, perturb(initial,301,16)[12],
                                             perturb(initial,1621,16)[4], perturb(initial,301,16)[7]])
            starts = np.tile(training_starts, (count,1))
            profiles = np.repeat(population,4,axis=0)
        else:
            profiles = population
        if args.standing_jump:
            starts[:,1] = -profiles[:,47]
        raw,sensors = replay(model,starts,profiles,runup,frames,True,keep_on_device=True)
        scores_blocks, passed_blocks, quality_blocks, detail = [], [], [], []
        for offset in range(0,len(starts),256):
            block = slice(offset,offset+256)
            raw_block = raw[block].contiguous().cpu().numpy()
            sensor_block = sensors[block].contiguous().cpu().numpy()
            measured, block_score, block_passed, block_detail = wall_assist(
                model,raw_block,sensor_block,initial,30,launch_window=85)
            block_score -= 6 * np.log1p(front_collision(model, sensor_block) / 30)
            scores_blocks.append(block_score)
            passed_blocks.append(block_passed)
            if args.balance:
                for row, (roll, delay) in zip(block_detail, balance_metrics(measured, block_detail)):
                    row.update(takeoff_to_front_contact_roll_deg=roll, front_touch_gap_s=delay)
            if args.launch_balance:
                for row, values in zip(block_detail, launch_metrics(model, measured, sensor_block, block_detail)):
                    row.update(values)
            if args.standing_forward:
                for row,values in zip(block_detail,launch_metrics(model,measured,sensor_block,block_detail)):
                    row.update(values)
            detail.extend(block_detail)
            if args.recovery:
                quality_blocks.append(landing_quality(measured))
        del raw, sensors
        scores, passed = np.concatenate(scores_blocks), np.concatenate(passed_blocks)
        if args.recovery:
            quality = np.concatenate(quality_blocks)
            scores += 16 * quality.mean(1) + 10 * quality[:,-25:].mean(1)
        if args.robust:
            stable = (quality[:,-25:] >= .99).all(1) if args.clear_rear else (
                np.lib.stride_tricks.sliding_window_view(quality >= .99,25,axis=1).all(-1).any(-1))
            # Rank by repeatability across phase and pose perturbations.
            objective = scores + 38 * stable + 12 * (quality[:,-25:] >= .99).mean(1)
            if args.clear_rear:
                force = np.array([row['forbidden_peak_n'] for row in detail])
                objective += 65 * passed - 18 * np.log1p(force / 15)
            if args.retain_stance:
                objective += 95 * stable + 95 * passed
            if args.balance:
                roll_cost = np.array([row['takeoff_to_front_contact_roll_deg']
                                      if row['takeoff_to_front_contact_roll_deg'] is not None else 100. for row in detail])
                gap_cost = np.array([row['front_touch_gap_s']
                                     if row['front_touch_gap_s'] is not None else .5 for row in detail])
                if args.balance_early:
                    objective -= 3.2*roll_cost + 380*gap_cost
                else:
                    objective -= .8*roll_cost + 105*gap_cost
            if args.launch_balance:
                objective -= np.array([1.4*row['launch_roll_deg'] + 55*abs(row['flight_lateral_m'])
                                       + 11*np.log1p(row['right_toe_wall_peak_n']/20)
                                       for row in detail])
                if args.toe_clearance:
                    objective -= np.array([24*np.log1p(row['right_toe_premature_wall_peak_n']/15)
                                           for row in detail])
                if args.roll_launch:
                    objective -= np.array([5*row['flight_peak_roll_deg'] + 3*abs(row['launch_roll_rate_deg_s'])/10
                                           for row in detail])
                if args.shape_push:
                    objective -= np.array([6*row['flight_peak_roll_deg'] + 3*row['launch_roll_deg']
                                           for row in detail])
            if args.clean_symmetric:
                objective = np.array([
                    160*bool(passed[i]) + 45*bool(stable[i])
                    + 36*row['best_far_foot_support_fraction']
                    + 10*row['far_ground_support_fraction']
                    - 18*np.log1p(row['forbidden_floor_peak_n']/12)
                    - 18*np.log1p(row['forbidden_wall_peak_n']/12)
                    - 4*row['flight_peak_roll_deg']
                    - 3*row['launch_roll_deg']
                    - (150*max(0.,.30-row['prelaunch_min_base_z_m']) if args.shallow_crouch else 0.)
                    - 75*max(0., row['flight_peak_roll_deg']-12.)/12.
                    for i,row in enumerate(detail)])
            if args.stabilize_shallow:
                objective += np.array([
                    130*bool(passed[i]) + 100*bool(stable[i])
                    - 650*abs(row['flight_lateral_m'])
                    - 90*max(0.,abs(row['flight_lateral_m'])-.065)
                    for i,row in enumerate(detail)])
            if args.protect_clearance:
                objective += np.array([
                    240*bool(passed[i]) - 105*(row['forbidden_peak_n'] >= 5.)
                    - 35*np.log1p(row['forbidden_peak_n']/15)
                    for i,row in enumerate(detail)])
            if args.micro_lateral:
                objective -= np.array([1750*abs(row['flight_lateral_m']) for row in detail])
            if args.push_lateral:
                objective -= np.array([2700*abs(row['flight_lateral_m']) for row in detail])
            if args.launch_timing:
                objective += np.array([
                    125*bool(passed[i]) - 35*(row['forbidden_wall_peak_n'] >= 5.)
                    + 22*np.clip(row.get('launch_forward_velocity_m_s',0.)/.9,0.,1.7)
                    - 40*max(0.,row.get('launch_distance_m',.5)-.30)
                    - 2*abs(row.get('launch_pitch_deg',35.)+12.)
                    for i,row in enumerate(detail)])
            if args.braking_recovery:
                objective += np.array([
                    210*bool(passed[i]) + 55*bool(stable[i])
                    - 115*(row['forbidden_wall_peak_n']>1000)
                    - 25*np.log1p(row['forbidden_floor_peak_n']/25)
                    for i,row in enumerate(detail)])
            if args.landing_sequence:
                objective = np.array([
                    290*bool(passed[i]) + 110*bool(quality[i,-25:].min()>=.99)
                    - 38*np.log1p(row['forbidden_wall_peak_n']/10)
                    - 38*np.log1p(row['forbidden_floor_peak_n']/10)
                    - 55*(row['forbidden_peak_n']>=5.)
                    for i,row in enumerate(detail)])
            if args.standing_jump:
                objective = np.array([
                    260*bool(passed[i]) + 75*bool(quality[i,-25:].min()>=.99)
                    + 55*row['best_far_foot_support_fraction'] + 20*row['flight_forward_m']
                    + 25*(row['rising_takeoff_frames'] is not None)
                    - 25*np.log1p(row['forbidden_peak_n']/15)
                    for i,row in enumerate(detail)])
            if args.standing_forward:
                objective += np.array([
                    90*np.clip(row.get('launch_forward_velocity_m_s',0.),-1.,2.)
                    + 70*np.clip(row['max_root_x_m'], -.8,.45)
                    - 1.7*abs(row.get('launch_pitch_deg',75.)+14.)
                    for row in detail])
            grouped = objective.reshape(count,4)
            ranking = grouped.mean(1) + .25 * grouped.min(1)
        else:
            ranking = scores
        i = int(np.argmax(ranking))
        print(iteration, float(ranking[i]), int(passed.sum()), detail[i * (4 if args.robust else 1)], flush=True)
        if ranking[i] > best:
            center = population[i]
            best = ranking[i]
            saved_initial = initial.copy()
            if args.standing_jump:
                saved_initial[1] = -center[47]
            np.savez_compressed(args.output/'best.npz',params=center,initial=saved_initial,runup=runup)
    if args.eval_only:
        with np.load(args.output / "best.npz") as saved:
            center = saved["params"].copy()
    report = {}
    profiles_to_eval = (('refined', center),) if args.standing_jump else (('baseline',prior),('refined',center))
    for version, profile in profiles_to_eval:
        if args.standing_jump:
            initial[1] = -profile[47]
        trials = {}
        for seed in (None,301,1621,*args.holdout):
            label = 'nominal' if seed is None else str(seed)
            starts = initial[None] if seed is None else perturb(initial,seed,16)
            raw,sensors = replay(model,starts,np.repeat(profile[None],len(starts),axis=0),runup,frames,False)
            m,score,passed,detail = wall_assist(model,raw,sensors,initial,30,launch_window=85)
            for row, force in zip(detail, front_collision(model, sensors)):
                row['front_wall_peak_n'] = float(force)
            for row, good in zip(detail, landing_quality(m) >= .99):
                edges = np.flatnonzero(np.diff(np.r_[False,good,False]))
                row['best_stable_seconds'] = float(np.max(edges[1::2]-edges[::2], initial=0) * .02)
            for row, (roll, delay) in zip(detail, balance_metrics(m,detail)):
                row.update(takeoff_to_front_contact_roll_deg=roll, front_touch_gap_s=delay)
            for row, values in zip(detail, launch_metrics(model,m,sensors,detail)):
                row.update(values)
            trials[label] = dict(passes=int(passed.sum()),trials=len(passed),detail=detail)
            if seed is None or seed == 301:
                np.savez_compressed(args.output/f'{version}-{label}.npz',states=m['state'][:1],
                                    air=m['air'][:1],success=passed[:1],scores=score[:1])
        report[version] = trials
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2))


def launch_metrics(model, measured, sensors, details):
    state = measured['state']
    contact = force_signals(model,sensors,'wall',FEET[:1])[...,0,:]
    right_wall = abs(contact[...,1])
    from .brace_experiment import signal
    toe = signal(model,sensors,'position_'+FEET[0])
    unsafe = (contact[...,6] < .288) | (abs(contact[...,9]) < .65) | (toe[...,0] < -.045)
    for i,row in enumerate(details):
        launch = row['rising_takeoff_frames']
        if launch is None:
            yield dict(launch_roll_deg=90., flight_lateral_m=1., right_toe_wall_peak_n=0., right_toe_premature_wall_peak_n=0., launch_yaw_deg=90., flight_yaw_deg=90., flight_peak_roll_deg=90., launch_roll_rate_deg_s=450., prelaunch_min_base_z_m=0.)
            continue
        a,b = launch
        w,x,y,z = state[i,a,3:7]
        roll = abs(np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))))
        w,x,y,z = state[i,a:b,3:7].T
        yaw = np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        yield dict(launch_roll_deg=float(roll),
                   launch_forward_velocity_m_s=float(state[i,a,19]),
                   launch_distance_m=float(-state[i,a,0]),
                   launch_pitch_deg=float(np.degrees(np.arcsin(np.clip(2*(state[i,a,3]*state[i,a,5]-state[i,a,6]*state[i,a,4]),-1,1)))),
                   launch_lateral_velocity_m_s=float(state[i,a,20]),
                   prelaunch_min_base_z_m=float(state[i,max(0,a-25):a+1,2].min()),
                   flight_peak_roll_deg=float(np.max(np.abs(np.degrees(np.arctan2(
                       2*(w*x+y*z),1-2*(x*x+y*y)))))),
                   launch_roll_rate_deg_s=float(np.degrees(state[i,a,22])),
                   launch_yaw_deg=float(np.degrees(yaw[0])),
                   flight_yaw_deg=float(np.degrees(yaw[-1]-yaw[0])),
                   flight_lateral_m=float(state[i,b-1,1]-state[i,a,1]),
                   right_toe_wall_peak_n=float(right_wall[i,a*SUBSTEPS:].max()),
                   right_toe_premature_wall_peak_n=float(np.where(unsafe[i],right_wall[i],0.)[a*SUBSTEPS:].max()))


def balance_metrics(measured, details):
    state, floor = measured['state'], measured['floor']
    w,x,y,z = np.moveaxis(state[...,3:7],-1,0)
    roll = np.abs(np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))))
    for i, row in enumerate(details):
        launch = row['rising_takeoff_frames']
        if launch is None:
            yield None, None
            continue
        left, right = [np.flatnonzero((floor[i,:,j] > 5.) & (state[i,:,0] > 0.)
                                      & (np.arange(state.shape[1]) >= launch[1])) for j in (3,2)]
        if len(left) and len(right):
            end = min(left[0],right[0])
            yield float(roll[i,launch[0]:end+1].max()), float(abs(left[0]-right[0])*.02)
        else:
            yield float(roll[i,launch[0]:launch[1]].max()), None


def landing_quality(m):
    far = (m['feet'][...,0].min(-1) > .064)
    upright = (m['up'] > .82) & (m['state'][...,2] > .22)
    support = (m['floor'] > 5.).mean(-1)
    slow = (m['speed'] < .35) & (m['spin'] < 2.)
    return far * upright * support * slow


def front_collision(model, sensors):
    links = tuple(f'{leg}_{part}_collision' for leg in ('FR','FL') for part in ('thigh','calf'))
    return abs(force_signals(model, sensors, 'wall',
                            FEET[2:] + links + ('base_link_collision',))[...,1]).max(axis=(1,2))


if __name__ == '__main__':
    main()
