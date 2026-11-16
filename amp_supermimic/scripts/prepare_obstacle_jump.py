import json,hashlib,subprocess,sys,shutil
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[2];D=R/'amp_supermimic/data';raw=D/'raw/robocon_jump_candidates'
raw.mkdir(exist_ok=True)
import urllib.request
base='https://huggingface.co/datasets/MIMUW-Robotics/kine2go/resolve/244dc58/'
for remote,local in [('data/solo8_jump_forward_b/traj_0001/traj.pkl','jump_b_0001.pkl'),('data/solo8_jump_forward_b/config.json','config.json')]:
 if not (raw/local).exists():urllib.request.urlretrieve(base+remote,raw/local)
x=torch.load(raw/'jump_b_0001.pkl',weights_only=True,map_location='cpu')
c=json.loads((raw/'config.json').read_text())
q=np.concatenate([v['dof_pos'].numpy() for v in x]);p=np.concatenate([v['base_pos'].numpy() for v in x]);quat=np.concatenate([v['base_quat'].numpy() for v in x])[:,[1,2,3,0]]
rot=Rotation.from_quat(quat);yaw=rot[0].as_euler('xyz')[2];align=Rotation.from_euler('z',-yaw)
p[:,:2]-=p[0,:2].copy();p=align.apply(p);quat=(align*rot).as_quat()
out=D/'go2/robocon_jump_candidates';out.mkdir(exist_ok=True)
f=out/'solo8_jump_b_rollout_0001.csv';np.savetxt(f,np.c_[p,quat,q],delimiter=',')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'source_fps':60,'joint_names':c['env_cfg']['dof_names'],'quaternion_order':'xyzw','files':[{'file':f.name,'sha256':sha(f),'source':'https://huggingface.co/datasets/MIMUW-Robotics/kine2go/blob/244dc58/data/solo8_jump_forward_b/traj_0001/traj.pkl','source_sha256':sha(raw/'jump_b_0001.pkl'),'frames':len(q),'status':'candidate_not_expert'}]}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(raw/'SOURCE.json').write_text(json.dumps(manifest,indent=2))
shutil.copy(D/'go2/kine2go_jump_rollouts/LICENSE',raw/'LICENSE')
assert all(int(x[i+1]['frame'].item())-int(x[i]['frame'].item()) == 1 for i in range(len(x)-1)), 'Reset/discontinuous frames'
for name,extra in [('baseline', ['--rotate-foot-calibration']),('neutral_ground',['--neutral-foot-calibration','--align-flat-ground'])]:
 dest=D/'wblwbl/obstacle_candidates'/name
 if dest.exists():
  print('Already exists, not overwriting:',dest);continue
 cmd=['/home/qyw/miniconda3/envs/trackerlab_amp/bin/python',str(R/'amp_supermimic/scripts/retarget_go2.py'),'--source',str(f),'--output',str(dest),'--swing-boost','0',*extra]
 with open('/tmp/retarget_'+name+'.log','w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 if (dest/'train.json').exists():(dest/'train.json').rename(dest/'candidate_manifest.json')
 print(name,json.loads((dest/'report.json').read_text()),flush=True)
