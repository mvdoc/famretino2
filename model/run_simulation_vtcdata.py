# coding: utf-8
"""
We have 5 simulations
1. Simple: using estimated parameters, increase units
2. Simple, fix gain: using estimated parameters, fix gain to 1., increase
   units
3. Test gain increase: fix gain of units to second identity to 1.,
   increase gain for first identity
4. Test pRF size increase: gain fixed to 1. for all units, increase pRF
   for first identity only
5. Test with more units: as in simulation 2., but with more units
"""

# # Run simulation using parameters estimated from Kay et al., 2015
# 
# In this notebook we'll be simulating what we think it's happening at the neural level to reduce the retinotopic bias found in the two behavioral experiments. We took the VTC data available from http://kendrickkay.net/vtcdata/, ran their CSS model (http://kendrickkay.net/socmodel/) using the same methods as in the paper, and stored the estimated parameters for every voxel that had a test performance greater than 50% (see their paper).
# 
# As in the original paper, the voxels are aggregated across participants for each ROI, and we also aggregate hemispheres. The simulation then randomly samples the voxels from this population of estimated parameters. For all simulations, we assume two population of neural units, each specifically responsive to a particular identity. We activate the populations when presenting the stimuli at the location of our experiments, and use these activity patterns to train a classifier to distinguish between the two identities. Then, we present morphed identities and use the classifier to distinguish between the identities at each stimulus location. Using the classifier's responses we build a psychometric fit, and store the PSE. 
# 
# In this way, we perform the following experiments:
# 
# 1. We increase the number of units responsive to the first identity, while keeping only one unit responsive to the second identity, using the estimated parameters
# 2. Same as 1., but we fix the gain for all voxels to $1.$
# 3. We keep the gain for the unit responsive to the second identity set to $1.$, and parametrically increase the gain for those units responsive to the first identity
# 4. We fix the gain for all units to $1.$, and we parametrically increase the receptive field size for the units responsive to the first identity
# 
# Experiments 3. and 4. are performed in order to check whether changes in unit parameters can also account for the reduced biases.

# Note that this notebook was used to prototype the final script. Please check the file `run_simulation_vtcdata.py` for the script.

import matplotlib
matplotlib.use('Agg')

import argparse
from cssmodel.simulation import *
import os.path as op
from joblib.parallel import Parallel, delayed


# define the scale from degrees to pixels
# we fitted the model with stimuli that lived in a 100x100 pix square
# and in the actual experiment the width was 12.5˚
# thus deg2pix = 100/12.5 = 8
# because our stimuli where centered at around 7˚, we are going to use a 200x200 square
# so they'll fit in the "screen"
deg2pix = 100/12.5
res = 200

# make stimuli
stimuli = make_stimuli(scale=deg2pix, res=res)


def compute_varpse(pses, avg=0.5):
    return np.sum((pses - avg)**2, axis=1)


def load_data(task=None):
    # laod data
    rois = ['V1', 'V2', 'V3', 'hV4', 'IOG', 'pFus', 'mFus']
    data = dict()
    data_combined = dict()
    for h in ['L', 'R']:
        for roi in rois:
            roi_name = f'{h}{roi}'
            # remove the center from x, y so they're centered in (0, 0)
            if task is None:
                datafn = f'../vtcdata/output/{roi_name}_median_param.txt'
            else:
                datafn = \
                    f'../vtcdata/output/{roi_name}_{task}task_median_param.txt'
            dt = np.loadtxt(datafn)
            # remove the center so we're agnostic of the center
            dt[:2] -= 50
            data[roi_name] = dt
    for roi in rois:
        data_combined[roi] = np.hstack((data[f'L{roi}'], data[f'R{roi}']))
    return data, data_combined


HERE = op.dirname(op.abspath(__file__))
OUTDIR = op.join(HERE, 'outputs')

SIMNAMES = [
    'increase_units',
    'increase_units_fixgain',
    'increase_units_increase_gain',
    'increase_units_increase_rfsize',
    'increase_units_increase_voxels'
]


def run_simulation(which, task=None, nproc=1, n_sim=500):
    if task is None:
        fnout = f'sim-{which:02d}_task-estimation_{SIMNAMES[which-1]}_{n_sim:03d}sim.csv'
    else:
        fnout = f'sim-{which:02d}_task-{task}_{SIMNAMES[which-1]}_{n_sim:03d}sim.csv'
    fnout = op.join(OUTDIR, fnout)

    if op.exists(fnout):
        raise ValueError(f'{fnout} exists, not overwriting')

    _, data = load_data(task=task)

    if which > 5:
        print(__doc__)
        raise ValueError('Simulations must be between 1 and 5')
    rois = [
        #'V1', 'V2', 'V3', 'hV4',
        'IOG', 'pFus', 'mFus'
    ]
    # setup various parameters
    ratios = np.arange(1, 10)

    # simulation 3: test change in gains
    gains = np.arange(1, 4.25, 0.5) if which == 3 else [1.]
    # simulation 4: increase rf size
    rf_increase = np.arange(0, 0.55, 0.1) if which == 4 else [0.]
    # simulation 5: increase number of units
    units_responsive_b = range(1, 6) if which == 5 else [1]
    # simulations 2-5: gain is fixed to 1.
    if which > 1:
        # fix gain
        data_fixgain = dict()
        for r, pp in data.items():
            p = pp.copy()
            p[3] = 1.
            data_fixgain[r] = p
        data = data_fixgain

    dfs = []
    for urb in units_responsive_b:
        n_voxels = (1 + ratios) * urb
        print(f"Using n_voxels: {n_voxels}")
        # set up combination of parameters
        ratio_nvox = [(r, ratio, nvx) for r in rois for (ratio, nvx)
                      in zip(ratios, n_voxels)]
        comb_search = [(r, ratio, nvx, g, rfi)
                       for g in gains
                       for rfi in rf_increase
                       for (r, ratio, nvx) in ratio_nvox]

        n_exps = len(comb_search)
        # create different master seeds for reproducibility
        rng = np.random.RandomState(245)
        master_seeds = [rng.randint(2**32) for _ in range(n_exps)]

        out = Parallel(n_jobs=nproc, verbose=5, backend='multiprocessing')(
            delayed(simulate_bunch_experiments_population)
            (data[r], stimuli=stimuli, ratio_voxels=ratio, n_voxels=nvx,
             scale_gain=g, increase_rf_size=rfi, n_sim=n_sim, sigma_noise=0.1,
             res=res) for (r, ratio, nvx, g, rfi), seed
            in zip(comb_search, master_seeds))

        _, pses, _, _ = zip(*out)
        varpses = [compute_varpse(p) for p in pses]
        df_varp = pd.DataFrame(varpses)
        df_varp.index = pd.MultiIndex.from_tuples(
            [(r, ratio, nvx, g, rfi)
             for (r, ratio, nvx, g, rfi) in comb_search],
            names=['roi', 'ratio', 'n_voxels', 'gain', 'rf_increase'])
        df_varp = df_varp.T
        df_varp = pd.melt(df_varp)
        df_varp['b_units'] = urb
        dfs.append(df_varp)
    dfs = pd.concat(dfs)

    dfs.to_csv(fnout)


def main():
    p = parse_args()
    run_simulation(p.simulation, p.task, p.nproc, p.nsim)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--simulation', type=int,
                        choices=range(1, 6),
                        help='which simulation to run',
                        required=True)
    parser.add_argument('--nsim', type=int,
                        help='number of simulation (default: 500)',
                        required=False, default=500)
    parser.add_argument('--task', type=str,
                        help='run on parameters estimated on a particular '
                             'task',
                        required=False, choices=['face'],
                        default=None)
    parser.add_argument('--nproc', type=int,
                        help='number of processes to use (default: 1)',
                        required=False, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    main()

