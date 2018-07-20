from numpy.testing import assert_almost_equal, assert_array_equal
from model import *


def test_sample_ecc():
    for _ in range(10):
        ecc = np.random.randint(100)
        x, y = sample_ecc(ecc)
        assert_almost_equal(x**2 + y**2, ecc**2)


def test_generate_params():
    cov = np.array([[0, 0], [0, 1]])
    rng = np.random.RandomState(32)
    params = generate_params(2, 2, cov, 10, rng=rng, scale=1.)
    rng_ = np.random.RandomState(32)
    params_ = generate_params(2, 2, cov, 10, rng=rng_, scale=1.)
    assert_array_equal(params, params_)
    assert params.shape == (10, 3)
    # check eccentricities
    xy = params[:, :2]
    assert_almost_equal(np.sum(xy ** 2, axis=1), [2**2] * 10)

    # check it does some sampling
    cov = np.eye(2)
    tol = 10**-9
    cov *= tol
    params = generate_params(2, 2, cov, 10, rng=rng, scale=1.)
    xy = params[:, :2]
    assert_almost_equal(np.sum(xy ** 2, axis=1), [2**2] * 10, decimal=3)
    assert_almost_equal(params[:, 2], [2] * 10, decimal=3)


def test_activation():
    data = np.zeros((100, 100))
    data[30, 30] = 1.
    # check that the gain is indeed different
    act_g1 = activation(data, 30, 30, 2, gain=1.0)
    act_g2 = activation(data, 30, 30, 2, gain=4.0)
    assert_almost_equal(act_g2/act_g1, 4.0)


def test_VoxelPopulation():
    n_voxels = 10
    n_stim = 4
    xs = ys = np.random.randn(n_voxels)*10 + 50
    sigmas = np.random.randn(n_voxels)*2
    pop = VoxelPopulation(xs, ys, sigmas, gain=2., n=0.5)

    gains = np.array([2.] * n_voxels)
    ns = np.array([0.5] * n_voxels)
    pop_v = VoxelPopulation(xs, ys, sigmas, gain=gains, n=ns)

    stim = np.zeros((n_stim, 100, 100))
    stim[0, 50, 50] = 1.

    act = pop.activate(stim)
    act_v = pop_v.activate(stim)
    assert act.shape == (n_stim, n_voxels)
    assert act_v.shape == (n_stim, n_voxels)
    assert_array_equal(act, act_v)
