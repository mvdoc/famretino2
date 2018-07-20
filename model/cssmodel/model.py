"""Module containing the implementation of the Compressive Spatial Summation
(CSS) model as described in Kay, K.N., Winawer, J., Mezer, A., & Wandell, B.A.
Compressive spatial summation in human visual cortex. Journal of
Neurophysiology (2013). See also http://kendrickkay.net/socmodel"""

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def gaussian2d(res, x0, y0, sx, sy):
    """
    Make a 2D gaussian

    Arguments
    ---------
    res : int
        size of one of the sides
    x0 : int
        column for center
    y0 : int
        row for center
    sx : float
        standard deviation for x
    sy : float
        standard deviation for y

    Returns
    -------
    gauss : np.array (res, res)
    """
    x = np.arange(0, res, 1, float)
    y = x[:, None]

    return np.exp(-(((x - x0) ** 2 / (2 * sx ** 2)) +
                    ((y - y0) ** 2 / (2 * sy ** 2))))


def ellipse(res, w, h, x0, y0):
    """
    Make a filled ellipse, parallel to the main axes

    Parameters
    ----------
    res : int
        size for one of the sides
    w : int
        width
    h : int
        height
    x0 : int
        column for center
    y0 : int
        row for center

    Returns
    -------
    ell : np.array (res, res)
    """
    x = np.arange(0, res, 1, float)
    y = x[:, None]
    rw = w/2.
    rh = h/2.
    return (((x - x0)**2/rh**2) + ((y - y0)**2/rw**2) <= 1.).astype(float)


def activation(data, x0, y0, sigma, gain=1.0, n=0.2, res=100):
    """
    Activation of one voxel/neuron according to the CSS model, which is
    defined as `gain * a**n`, where `a = sum(data*gaussian)`.

    Parameters
    ----------
    data : np.array (n_stimuli, res, res)
        the data containing the stimuli
    x0 : int
        column for center of the receptive field
    y0 : int
        row for center of the receptive field
    sigma : float
        standard deviation of the gaussian; receptive field size is defined
        as sigma/sqrt(n)
    gain : float
        the gain
    n : float
        the power exponent
    res : size for one of the sides

    Returns
    -------
    activation : float
        the resulting activation
    """
    # make integral equal to 1.
    gauss = gaussian2d(res, x0, y0, sigma, sigma) / (2*np.pi*sigma**2)
    if data.ndim == 2:
        data = data[None]
    return gain * (data * gauss[None]).sum(axis=(1, 2)) ** n


def sample_ecc(ecc, n=1, rng=None):
    """
    Given an eccentricity value, return coordinates of a random center with
    given eccentricity

    Parameters
    ----------
    ecc : float
        eccentricity
    n : int
        number of centers to return
    rng : np.RandomState

    Returns
    -------
    center : np.array (2,)
        the x, y coordinates
    """
    if rng is None:
        rng = np.random.RandomState()
    # sample angles
    theta = rng.rand(n) * 2 * np.pi
    x = ecc * np.cos(theta)
    y = ecc * np.sin(theta)
    return np.hstack((x, y))


def generate_params(ecc, prf_sigma, cov, n_voxels, scale=4., rng=None):
    """
    Generate parameters for a population of voxels given
    mean eccentricity, mean RF size, and covariance between eccentricity
    and RF size

    Parameters
    ----------
    ecc : float
        average eccentricity
    prf_sigma : float
        standard deviation of the pRF (this is NOT normalized by sqrt(n))
    cov : np.array (2, 2)
        covariance between ecc and prf_sigma
    n_voxels : int
        number of voxels to generate
    scale : float
        scaling factor to use. `scale` pixels = 1 deg
    rng : numpy.RandomState or None
        random number generator for reproducibility

    Returns
    -------
    xys : array (n_voxels, 3)
        parameters for the generated n_voxels pRFs with
        center (xys[i, 0], xys[i, 1]) and sd (xys[i, 2])
    """
    if rng is None:
        rng = np.random.RandomState()
    # from Figure S2B
    centers = np.array((ecc, prf_sigma)) * scale
    cov = np.asarray(cov) * scale
    nn = np.abs(rng.multivariate_normal(centers, cov, n_voxels))
    xys = []
    for ecc, ss in nn:
        # get random coordinates given eccentricity
        x, y = sample_ecc(ecc, rng=rng)
        xys_ = np.hstack((x, y, ss))
        xys.append(xys_)
    xys = np.array(xys)
    return xys


class VoxelPopulation(object):
    """A population of voxels"""
    def __init__(self, xs, ys, sigmas, gain=1.0, n=0.2, res=100):
        """
        Initialize a population of voxels with centers (xs, ys) and
        receptive field sizes sigmas.

        Arguments
        ---------
        xs : array (n_voxels,)
            these are assumed to be centered in (0, 0)
        ys : array (n_voxels,)
            these are assumed to be centered in (0, 0)
        sigmas : array (n_voxels,)
        gain : float or array (n_voxels, )
            gain to use in CSS model (default 1.0)
        n : float or array (n_voxels, )
            exponent for compressive summation (default 0.2)
        res : int
            resolution (default 100)
        """
        x0 = y0 = res // 2
        self.xs = np.asarray(xs) + x0
        self.ys = np.asarray(ys) + y0
        self.sigmas = sigmas
        self.gain = np.array([gain] * len(xs)) if isinstance(gain, float) \
            else gain
        self.n = np.array([n] * len(xs)) if isinstance(n, float) else n
        self.res = res
        self.n_voxels = len(self.xs)

    def activate(self, stim):
        """
        Compute response of the population when presented with stim

        Parameters
        ----------
        stim : array (n_stimuli, res, res)

        Returns
        -------
        activations : array (n_stimuli, n_voxels)
            the activations for each of the stimuli
        """
        act = []
        for x, y, s, g, n in zip(self.xs, self.ys, self.sigmas, self.gain,
                                 self.n):
            act.append(activation(stim, x, y, s, gain=g, n=n,
                                  res=self.res))
        return np.stack(act).T


def make_stimuli(w=4, h=4, xpos=(5, -5, -5, 5), ypos=(5, 5, -5, -5),
                 scale=4., res=100):
    """
    Make stimuli for retinotopic experiment (use contrast aperture)

    Parameters
    ----------
    w : float
        width of the ellipse
    h : float
        height of the ellipse
    xpos : array (n_stims, )
        x center of the stimuli
    ypos : array (n_stims, )
        y center of the stimuli
    scale : float
        scaling factor. `scale` pixels = 1 deg
    res : int
        resolution

    Returns
    -------
    stimuli : array (n_stims, res, res)
        well, the stimuli
    """
    w = int(scale * w)
    h = int(scale * h)
    x0 = y0 = res // 2
    xpos = scale * np.asarray(xpos) + x0
    ypos = scale * np.asarray(ypos) + y0

    stims = []
    for x, y in zip(xpos, ypos):
        stims.append(ellipse(res, w, h,  x, y))
    return np.array(stims)


def plot_prfs(xs, ys, sigmas, ax=None, res=100, n=0.2):
    """
    Plot individual receptive fields as contours. The countours indicate a
    radius of 2sigma.

    Parameters
    ----------
    xs : array (n_voxels,)
    ys : array (n_voxels, )
    sigmas : array (n_voxels, )
    ax : axis
    res : int
    n : float or array (n_voxels, )

    Returns
    -------
    ax : axis containing the image

    """
    x0 = y0 = res // 2
    xs = np.asarray(xs) + x0
    ys = np.asarray(ys) + y0
    if isinstance(n, float):
        n = np.array([n] * len(xs))
    # 4. is 2. for diameters multiplied by 2. for 2sigma
    # normalize by sqrt(n) as in Kay et al., 2015
    widths = 4. * np.asarray(sigmas) / np.sqrt(n)

    ells = [Ellipse(xy=[x, y], width=w, height=w,
                    facecolor='none', edgecolor='black')
            for x, y, w in zip(xs, ys, widths)]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
    ax.set_xlim(0, res)
    ax.set_ylim(0, res)
    ax.set_aspect('equal')
    # draw points as well
    ax.scatter(xs, ys, c='red', s=3)
    ax.invert_yaxis()
    ax.axhline(y0)
    ax.axvline(x0)
    return ax


def plot_gauss(xs, ys, sigmas, ax=None, res=100):
    """
    Plot sum of gaussians defined by centers (xs, ys) and sigma

    Parameters
    ----------
    xs : array (n_voxels,)
    ys : array (n_voxels, )
    sigmas : array (n_voxels, )
    ax : axis
    res : int

    Returns
    -------
    ax : axis
    """
    x0 = y0 = res // 2
    xs = np.asarray(xs) + x0
    ys = np.asarray(ys) + y0

    gauss = []
    for x, y, s in zip(xs, ys, sigmas):
        gauss.append(gaussian2d(res, x, y, s, s))
    gauss = np.stack(gauss)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.imshow(gauss.sum(axis=0), origin='upper')
    ax.axhline(y0)
    ax.axvline(x0)
    ax.set_aspect('equal')

    return ax


def filledprf(x0, y0, sigma, n=0.2, res=100):
    """
    Fill the pRF centered in x0, y0 and radius sigma

    Arguments
    ---------
    x0 : int
        column for center
    y0 : int
        row for center
    sigma : float
        standard deviation of the pRF
    n : float
        power exponent
    res : int
        size of one of the sides

    Returns
    -------
    prf : array (res, res)
        the filled pRF
    """
    x = np.arange(0, res, 1, float)
    y = x[:, None]
    s = 2 * sigma / np.sqrt(n)
    return ((x - x0) ** 2 + (y - y0) ** 2 <= s ** 2).astype(int)


def plot_prfdensity(xs, ys, sigmas, n=0.2, res=100, ax=None):
    """
    Plot the coverage of a population of voxels

    Parameters
    ----------
    xs : array (n_voxels)
        centers
    ys : array (n_voxels)
        centers
    sigmas : array (n_voxels)
        std pRfs
    n : float or array (n_voxels)
        power exponent
    res : int
        size of one of the sides
    ax : axis

    Returns
    -------
    ax : axis
    """
    x0 = y0 = res // 2
    xs = np.asarray(xs) + x0
    ys = np.asarray(ys) + y0

    if isinstance(n, float):
        n = np.array([n] * len(xs))

    f = np.zeros((res, res))
    for x, y, s, n_ in zip(xs, ys, sigmas, n):
        f += filledprf(x, y, s, n=n_, res=res)
    f /= len(xs)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.imshow(f, aspect='equal', cmap='gray')
    ax.axhline(y0)
    ax.axvline(x0)
    return ax
