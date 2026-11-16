"""Build WBL direction reference candidates from the LVR Isaac Lab traces."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation

REVISION = "e68ab0622c550e0809f77af707cd7b2fefd43dbb"
FPS = 50
SOURCE = "https://github.com/ruipengZ/Latent-Variation-Regularization"


def reconstruct(obs, names):
    neutral = [-.1 if n[1]=='R' else .1 for n in names]
    neutral = np.array([angle if 'hip' in n else
                        (.8 if n[0]=='F' else 1.) if 'thigh' in n else -1.5
                        for n,angle in zip(names,neutral)])
    gravity = obs[:,6:9]
    roll = np.arctan2(-gravity[:,1],-gravity[:,2])
    pitch = np.arctan2(gravity[:,0],np.linalg.norm(gravity[:,1:],axis=-1))
    tilt = Rotation.from_euler('xyz', np.c_[roll,pitch,np.zeros(len(obs))])
    yaw = cumulative_trapezoid(tilt.apply(obs[:,3:6]/.2)[:,2], dx=1/FPS, initial=0)
    rotation = Rotation.from_euler('z', yaw)*tilt
    pos = cumulative_trapezoid(rotation.apply(obs[:,:3]), dx=1/FPS, axis=0, initial=0)
    # Absolute pose is absent. Remove vertical integration drift and choose a nominal initial height.
    pos[:,2] += .33 - np.linspace(0,pos[-1,2],len(pos))
    return np.c_[pos,rotation.as_quat(),obs[:,12:24]+neutral]


def mirror(rows, names):
    result=rows.copy()
    result[:,1]*=-1
    result[:,[3,5]]*=-1
    for i,name in enumerate(names):
        other=name[0]+('R' if name[1]=='L' else 'L')+name[2:]
        result[:,7+i]=rows[:,7+names.index(other)]*(-1 if 'hip' in name else 1)
    return result


def main():
    import h5py

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--joint-order', type=Path, required=True, help='Source observation joint order in JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retarget-python', default=sys.executable)
    parser.add_argument('--motion-scale', type=float, default=1.0)
    args=parser.parse_args()
    data=args.repo/'amp_supermimic/data'
    source=data/'go2/lvr_velocity_demos'
    out=args.output
    names=json.loads(args.joint_order.read_text())
    assert len(names)==12 and set(names)=={f'{leg}_{joint}_joint' for leg in ('FR','FL','RR','RL') for joint in ('hip','thigh','calf')}
    go=out/'go2'
    go.mkdir(parents=True,exist_ok=False)
    entries=[]
    for task, tag, axis, sign in [('backward','backward',0,-1),('sideway','left',1,1)]:
        path=source/f'isaac_go2_{task}_traj-1_compact.hdf5'
        with h5py.File(path) as f:
            obs=f['_meta/obs'][50:300]
            assert obs.shape==(250,48) and np.isfinite(obs).all()
            assert not f['_meta/done'][50:299].any()
        assert np.median(obs[:,axis]*sign)>.2
        rows=reconstruct(obs,names)
        variants={tag:rows}
        if tag=='left':
            variants['right']=mirror(rows,names)
            assert np.allclose(mirror(variants['right'],names),rows)
        for variant, values in variants.items():
            csv=go/f'lvr_{variant}.csv'
            np.savetxt(csv,values,delimiter=',',fmt='%.9g')
            entries.append(dict(file=csv.name,sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                                source=str(path.relative_to(args.repo)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                rows=[50,300], mirrored=variant=='right',
                                donor_body_velocity_m_s=(np.median(obs[:,:3],axis=0)*[1,-1 if variant=='right' else 1,1]).tolist()))
    (go/'manifest.json').write_text(json.dumps(dict(source_fps=FPS,quaternion_order='xyzw',joint_names=names,
                                               source_repository=SOURCE,revision=REVISION,
                                               root_pose='reconstructed from velocity and gravity, absolute position not recorded',
                                               files=entries),indent=2)+'\n')
    for entry in entries:
        csv=go/entry['file']
        dest=out/'wblwbl'/csv.stem
        subprocess.run([args.retarget_python,'-B',str(args.repo/'amp_supermimic/scripts/retarget_go2.py'),
                        '--source',str(csv),'--output',str(dest),'--neutral-foot-calibration',
                        '--align-flat-ground','--swing-boost','0','--motion-scale',
                        str(args.motion_scale)],check=True)
        manifest=dest/'train.json'
        if manifest.exists():
            manifest.rename(dest/'candidate_manifest.json')


if __name__=='__main__':
    main()
