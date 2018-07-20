from .simulation import *
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_sample_from_population():
    n_voxels = 10
    params = np.random.randn(5, n_voxels)

    p, vp = sample_from_population(params, n_voxels=n_voxels,
                                   ratio_voxels=n_voxels-1)
    assert p['a'].shape == (5, n_voxels - 1)
    assert p['b'].shape == (5, 1)

    # set gain to 1.
    params[3] = 1.

    # check change in gain works
    p, vp = sample_from_population(params, n_voxels=n_voxels, scale_gain=3.,
                                   ratio_voxels=n_voxels-1)

    assert_array_equal(p['a'][3], 3. * np.ones(n_voxels - 1))
    assert_array_equal(p['b'][3], [1.])

    # check we're sampling without replacement
    p, vp = sample_from_population(params, n_voxels=n_voxels, scale_gain=1.,
                                   increase_rf_size=0,
                                   ratio_voxels=n_voxels)
    # with ratio_voxels set to n_voxels all voxels are in a
    # resort both params and p['a'] in the same way
    params = params[:, np.argsort(params[0])]
    p['a'] = p['a'][:, np.argsort(p['a'][0])]

    assert_array_equal(params, p['a'])

    # check increase in rf size works
    p, vp = sample_from_population(params, n_voxels=n_voxels, scale_gain=1.,
                                   increase_rf_size=0.10,
                                   ratio_voxels=n_voxels)
    # with ratio_voxels set to n_voxels all voxels are in a
    # resort both params and p['a'] in the same way
    params = params[:, np.argsort(params[0])]
    p['a'] = p['a'][:, np.argsort(p['a'][0])]

    prf = params[2]
    prf_in = p['a'][2]
    assert_array_almost_equal((prf_in - prf)/prf, 0.10 * np.ones(n_voxels))
