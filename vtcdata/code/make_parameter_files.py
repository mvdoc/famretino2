import numpy as np
import os
import os.path as op
from scipy.io import loadmat

rois = ['V1', 'V2', 'V3', 'hV4', 'IOG', 'pFus', 'mFus']

def filter_voxels(res, cutoff=50):
    # as in Kay et al., select non-noisy voxels with at least 50% variance explained
    idx = res['aggregatedtestperformance'][0] >= cutoff
    return np.median(res['params'][..., idx], axis=0)

HERE = op.dirname(op.abspath(__file__))
OUTDIR = op.join(op.dirname(HERE), 'output')

params = dict()
for hemi in ['L', 'R']:
    for roi in rois:
        ok_voxs = []
        for s in range(1, 4):
            res = loadmat(op.join(OUTDIR, f'sub-{s:02d}_{hemi}{roi}.mat'))
            ok_voxs.append(filter_voxels(res))
        ok_voxs = np.hstack(ok_voxs)
        params[f'{hemi}{roi}'] = ok_voxs

# save parameters for later use
header = ['row', 'col', 'std', 'gain', 'n']

for roi, param in params.items():
    fnout = op.join(OUTDIR, f'{roi}_median_param.txt')
    if not op.exists(fnout):
        np.savetxt(fnout, param, header=' '.join(header))
    else:
        print(f'Skipping {fnout}, file exists')

