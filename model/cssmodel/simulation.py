"""Functions to run a simulation"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from .model import *


def get_params_pop(mu_ecc, mu_size, cov, n_voxels=10, ratio_voxels=1.,
                   ratio_gain=1., scale=4., rng=None):
    """
    Return parameters for a population of voxels coding for two identities.
    We're assuming that the population has a total number of voxels
    `n_voxels`, and that the ratio of voxels being selective for a vs. b is
    `ratio_voxels`, and the ratio of gain a/b is `ratio_gain`.

    Arguments
    ---------
    mu_ecc : float
        average eccentricity
    mu_size : float
        average size
    cov : array (2, 2)
        covariance matrix between eccentricity and size
    n_voxels : int
        total number of voxels
    ratio_voxels : float
        proportion of voxels coding for one identity vs. another
    ratio_gain : float
        ratio of gain for identity a vs. identity b. Gain for identity b is
        always set to 1.0
    scale : float
        scaling factor (mostly for plotting)
    rng : numpy.RandomState
        random state

    Returns
    -------
    params : dict
        maps stim -> array (n_voxels, 3) representing x, y, sd size
    voxel_pops : dict
        maps stim -> VoxelPopulation
    """
    if rng is None:
        rng = np.random.RandomState()

    stims = ['a', 'b']
    #     n_voxels_pop = {
    #         'a': np.ceil(n_voxels*ratio_voxels).astype(int),
    #         'b': n_voxels
    #     }
    # let's figure out the number of voxels
    b = int(n_voxels // (1 + ratio_voxels))
    n_voxels_pop = {
        'a': n_voxels - b,
        'b': b
    }

    gain_pop = {
        'a': ratio_gain,
        'b': 1.
    }

    # get parameters
    params = {
        s: generate_params(mu_ecc, mu_size, cov, n_voxels=n_voxels_pop[s],
                           scale=scale, rng=rng).T
        for s in stims
    }
    # get voxel populations
    voxel_pops = {
        s: VoxelPopulation(xs, ys, sigmas, gain=gain_pop[s])
        for s, (xs, ys, sigmas) in params.items()
    }
    return params, voxel_pops


def generate_activations(act_a, act_b, n_trials_stim, weights=(1, 0),
                         sigma_noise=0.05, rng=None):
    """
    Generate noisy population activations based on non-noisy activations.

    Parameters
    ----------
    act_a : array (n_stimuli, n_voxels_a)
        the non-noisy activation response of population a to the stimuli
    act_b : array (n_stimuli, n_voxels_b)
        the non-noisy activation response of population b to the stimuli
    n_trials_stim : int
        the number of trials to simulate for each of the stimuli
    weights : array-like (2,)
        weights for the activation; the final activation will be obtained by
        stacking  `weights[0] * act_a` and `weights[1] * act_b`. This is to
        simulate mixed selectivity responses, or alternatively the
        presentation of a morphed stimulus between `a` and `b`.
    sigma_noise : float
    rng : random state

    Returns
    -------
    activations : array (n_stimuli * n_trials_stim, n_voxels_a + n_voxels_b)
        the population responses
    stimuli : array (n_stimuli * n_trials_stim,)
        array indexing the stimulus number
    """
    # figure out how much data we need to create
    n_voxels_a = act_a.shape[1]
    n_voxels_b = act_b.shape[1]
    n_voxels = n_voxels_a + n_voxels_b
    n_stims = act_a.shape[0]
    assert n_stims == act_b.shape[0]
    # preallocate some nice noise
    if rng is None:
        rng = np.random.RandomState()
    acts = rng.randn(n_stims * n_trials_stim, n_voxels) * sigma_noise
    # weight the response
    wa, wb = weights
    w_response = np.hstack((wa * act_a, wb * act_b))
    acts += np.repeat(w_response, n_trials_stim, axis=0)
    stimuli = np.repeat(np.arange(n_stims), n_trials_stim)
    return acts, stimuli


def get_possible_ratios(n_voxels):
    """
    Given a total number of voxels, return possible ratios

    Parameters
    ----------
    n_voxels

    Returns
    -------
    ratios : array (n_ratios)
    """
    ok_voxels = np.arange(1, n_voxels//2 + 1)
    return (n_voxels - ok_voxels)/ok_voxels.astype(float)


def simulate_experiment(voxel_pops, stimuli, n_trials_stim=10,
                        morphs=np.linspace(0, 1, 7), sigma_noise=0.05,
                        rng=None):
    """

    Parameters
    ----------
    voxel_pops : dict
        dictionary mapping identity label to an instance of VoxelPopulation
    stimuli : array (n_stimuli, res, res)
        the stimuli
    n_trials_stim : int
        number of trials for each stimulus used in the experiment
    morphs : array (n_morphs,)
        morphing values to build the psychometric curve
    sigma_noise : float
    rng : RandomState

    Returns
    -------
    df : pd.DataFrame
        dataframe containing the simulated responses
    training_score : float
        training score for the classifier
    """
    # STEP 1. Generate activations from voxel populations
    # compute activations without noise for each population
    activations = dict()
    for s, vp in voxel_pops.items():
        activations[s] = vp.activate(stimuli)

    # generate training set
    act_a = activations['a']
    act_b = activations['b']
    # only response to a
    response_a, _ = generate_activations(act_a, act_b, n_trials_stim,
                                         weights=[1, 0],
                                         sigma_noise=sigma_noise,
                                         rng=rng)
    # only response to b
    response_b, _ = generate_activations(act_a, act_b, n_trials_stim,
                                         weights=[0, 1],
                                         sigma_noise=sigma_noise,
                                         rng=rng)
    # put all together
    responses = np.vstack((response_a, response_b))

    # STEP 2. Have a classifier learn to distinguish based on the responses
    # now it's time to learn
    n_stimuli = len(stimuli)
    targets = np.repeat([0, 1], n_trials_stim * n_stimuli)
    svc = SVC(kernel='linear')
    svc.fit(responses, targets)
    training_score = svc.score(responses, targets)

    # STEP 3. Simulate experiment with different morphings in different
    # locations
    responses_exp = []
    for m in morphs:
        responses_exp.append(
            generate_activations(act_a, act_b, n_trials_stim,
                                 weights=[1. - m, m],
                                 sigma_noise=sigma_noise, rng=rng)[0]
        )
    responses_exp = np.stack(responses_exp)

    # now predict for each morph
    resp = []
    loc = []
    morph_lbl = []
    for m, r in zip(morphs, responses_exp):
        p = svc.predict(r)
        resp.extend(p)
        loc.extend(np.repeat(np.arange(n_stimuli), n_trials_stim))
        morph_lbl.extend(np.repeat(np.round(m, 2), len(p)))

    # Store the "responses" in a dataframe
    df = pd.DataFrame(
        {
            'location': loc,
            'resp': resp,
            'morph': morph_lbl
        },
        columns=['location', 'morph', 'resp'])
    df['location'] = pd.Categorical(df['location'])

    return df, training_score


def compute_pse(df, formula='resp ~ morph + location - 1'):
    """
    Compute Point of Subjective Equality based on responses in df. This is
    done by fitting a logit model on the response, and finding the point of
    inflection.

    Parameters
    ----------
    df : pd.DataFrame
        each row is a trial
    formula : str
        formula passed to the glm for fitting

    Returns
    -------
    pse : array (n_locations, )
        the pse estimates for each location
    """
    model = smf.glm(formula, df, family=sm.families.Binomial())
    try:
        res = model.fit()
        # now return the estimates
        p = res.params.as_matrix()
        pse = -p[:-1] / p[-1]
    except PerfectSeparationError:
        print("WARNING: got PerfectSeparationError, filling pses with 0.5")
        pse = np.ones(len(df.location.unique())) * 0.5

    return pse


def simulate_bunch_experiments(mu_ecc, mu_size, cov, stimuli, n_sim=100,
                               n_voxels=10, ratio_voxels=1.0,
                               ratio_gain=1.0, sigma_noise=0.05,
                               master_seed=234):
    """
    Simulate a bunch of experiments

    Parameters
    ----------
    mu_ecc : float
        average eccentricity value
    mu_size : float
        average size of pRF
    cov : array (2, 2)
        covariance between eccentricity and size
    stimuli : array (n_stimuli, res, res)
        the stimuli used
    n_sim : int
        number of simulations to run
    n_voxels : int
        number of voxels
    ratio_voxels : float
        ratio of voxels responsive to one identity
    ratio_gain : float
        ratio for gains responsive to one identity
    sigma_noise : float
    master_seed : int
        master seed used for reproducibility

    Returns
    -------
    df : pd.DataFrame
        dataframe containing all the simulations
    pses : array (n_sim, n_stimuli)
        the estimated pses for each simulation
    training_scores : array (n_sim)
        training scores for each simulation
    """
    master_rng = np.random.RandomState(master_seed)
    rngs = [np.random.RandomState(r)
            for r in master_rng.randint(2**32, size=n_sim)]

    df_sim = []
    training_scores = []
    pses = []
    for i, rng in enumerate(rngs):
        params, voxel_pops = get_params_pop(mu_ecc, mu_size, cov,
                                            n_voxels=n_voxels,
                                            ratio_voxels=ratio_voxels,
                                            ratio_gain=ratio_gain,
                                            rng=rng)
        df, ts = simulate_experiment(voxel_pops, stimuli,
                                     sigma_noise=sigma_noise)
        pse = compute_pse(df)
        df['simulation'] = i
        df_sim.append(df)
        training_scores.append(ts)
        pses.append(pse)
    return pd.concat(df_sim), np.vstack(pses), np.asarray(training_scores)


def sample_from_population(roi_params, n_voxels=10, ratio_voxels=1.,
                           scale_gain=1., increase_rf_size=0.,
                           res=100, rng=None):
    """
    Return parameters for a population of voxels coding for two identities.
    We're assuming that the population has a total number of voxels
    `n_voxels`, and that the ratio of voxels being selective for a vs. b is
    `ratio_voxels`, and the ratio of gain a/b is `ratio_gain`.

    Arguments
    ---------
    roi_params : array (5, n_voxels)
        array containing the five parameters to sample, that is
        params[0]: row index of pRF center (center must be (0, 0))
        params[1]: column index of pRF center (center must be (0, 0))
        params[2]: standard deviation of gaussian (not normalized by n)
        params[3]: gain parameter
        params[4]: exponent of power-law non-linearity
    n_voxels : int
        total number of voxels
    ratio_voxels : float
        proportion of voxels coding for one identity vs. another
    scale_gain : float
        this will be multiplied to the gains of the parameters for the
        first identity. if all gains are set to 1., this is equivalent to
        increasing the ratio of gains
    increase_rf_size : float
        percentage of increase of receptive field size for identity a. For
        example, 0.05 corresponds to a 5% increase in receptive field size.
        0. means no increase
    res : int
        width of the image in pixel size
    rng : numpy.RandomState
        random state

    Returns
    -------
    params : dict
        maps stim -> array (n_voxels, 5) representing x, y, sd size, gain, n
    voxel_pops : dict
        maps stim -> VoxelPopulation
    """
    if rng is None:
        rng = np.random.RandomState()

    stims = ['a', 'b']
    # let's figure out the number of voxels
    b = int(n_voxels // (1 + ratio_voxels))
    n_voxels_pop = {
        'a': n_voxels - b,
        'b': b
    }

    n_voxels_roi = roi_params.shape[1]
    params = {
        s: roi_params[:, rng.choice(range(n_voxels_roi), n_voxels_pop[s],
                                    replace=False)]
        for s in stims
    }
    # scale the gains
    params['a'][3] *= scale_gain

    # increase RF size
    params['a'][2] += params['a'][2] * increase_rf_size

    # get voxel populations
    voxel_pops = {
        s: VoxelPopulation(xs, ys, sigmas, gain=gains, n=ns, res=res)
        for s, (xs, ys, sigmas, gains, ns) in params.items()
    }
    return params, voxel_pops


def simulate_bunch_experiments_population(
        roi_params, stimuli, n_sim=100, n_voxels=10, ratio_voxels=1.0,
        scale_gain=1.0, increase_rf_size=0., sigma_noise=0.05,
        res=100, master_seed=234):
    """
    Simulate a bunch of experiments sampling from an array of parameters

    Parameters
    ----------
    roi_params : array (5, n_voxels)
        array containing the five parameters to sample, that is
        params[0]: row index of pRF center (center must be (0, 0))
        params[1]: column index of pRF center (center must be (0, 0))
        params[2]: standard deviation of gaussian (not normalized by n)
        params[3]: gain parameter
        params[4]: exponent of power-law non-linearity
    stimuli : array (n_stimuli, res, res)
        the stimuli used
    n_sim : int
        number of simulations to run
    n_voxels : int
        number of voxels
    ratio_voxels : float
        ratio of voxels responsive to one identity
    scale_gain : float
        this will be multiplied to the gains of the parameters for the
        first identity. if all gains are set to 1., this is equivalent to
        increasing the ratio of gains
    increase_rf_size : float
        percentage of increase of receptive field size for identity a. For
        example, 0.05 corresponds to a 5% increase in receptive field size.
        0. means no increase
    sigma_noise : float
    res : int
        width of the image in pixel size
    master_seed : int
        master seed used for reproducibility

    Returns
    -------
    df : pd.DataFrame
        dataframe containing all the simulations
    pses : array (n_sim, n_stimuli)
        the estimated pses for each simulation
    training_scores : array (n_sim)
        training scores for each simulation
    parameters : list of dict
        parameters used for every simulation
    """
    master_rng = np.random.RandomState(master_seed)
    rngs = [np.random.RandomState(r)
            for r in master_rng.randint(2**32, size=n_sim)]

    df_sim = []
    training_scores = []
    pses = []
    parameters = []
    for i, rng in enumerate(rngs):
        params, voxel_pops = sample_from_population(
            roi_params, n_voxels=n_voxels, scale_gain=scale_gain,
            increase_rf_size=increase_rf_size, ratio_voxels=ratio_voxels,
            res=res, rng=rng)
        df, ts = simulate_experiment(voxel_pops, stimuli,
                                     sigma_noise=sigma_noise)
        pse = compute_pse(df)
        df['simulation'] = i
        df_sim.append(df)
        training_scores.append(ts)
        pses.append(pse)
        parameters.append(params)
    return (pd.concat(df_sim), np.vstack(pses), np.asarray(training_scores),
            parameters)
