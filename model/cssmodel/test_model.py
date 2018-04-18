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