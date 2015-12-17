import numpy as np
from fatiando import mesher, gridder
from fatiando.gravmag import prism
from numpy.testing import assert_array_almost_equal as assert_almost
from nose.tools import assert_raises
from nose.tools import assert_equal
import scipy.optimize
from numpy.random import RandomState


def test_fails_if_bad_pad_operation():
    'gridder.pad_array fails if given a bad padding array operation option'
    p = 'foo'
    shape = (100, 100)
    x, y, z = gridder.regular((-1000., 1000., -1000., 1000.), shape, z=-150)
    g = z.reshape(shape)
    assert_raises(ValueError, gridder.pad_array, g, padtype=p)


def test_pad_and_unpad_equal_2d():
    'gridder.pad_array and subsequent .unpad_array gives original array: 2D'
    shape = (100, 101)
    x, y, z = gridder.regular((-5000., 5000., -5000., 5000.), shape, z=-150)
    # rosenbrock: (a-x)^2 + b(y-x^2)^2  a=1 b=100 usually
    X = x.reshape(shape)
    Y = y.reshape(shape)
    xy = [x, y]
    gz = scipy.optimize.rosen([Y/100000., X/100000.])
    pads = ['mean', 'edge', 'lintaper', 'reflection', 'oddreflection',
            'oddreflectiontaper', '0']
    for p in pads:
        gpad, nps = gridder.pad_array(gz, padtype=p)
        gunpad = gridder.unpad_array(gpad, nps)
        assert_almost(gunpad, gz)


def test_pad_and_unpad_equal_1d():
    'gridder.pad_array and subsequent .unpad_array gives original array: 1D'
    prng = RandomState(12345)
    x = prng.rand(21)
    xpad, nps = gridder.pad_array(x)
    xunpad = gridder.unpad_array(xpad, nps)
    assert_almost(xunpad, x)


def test_coordinatevec_padding_1d():
    'gridder.padcoords accurately pads coordinate vector in 1D'
    prng = RandomState(12345)
    f = prng.rand(72) * 10
    x = np.arange(100, 172)
    fpad, nps = gridder.pad_array(f)
    xpad = gridder.pad_coords(x, f.shape, nps)
    assert_almost(xpad[0][nps[0][0]:-nps[0][1]], x)


def test_coordinatevec_padding_2d():
    'gridder.padcoords accurately pads coordinate vector in 2D'
    shape = (101, 172)
    x, y, z = gridder.regular((-5000., 5000., -5000., 5000.), shape, z=-150)
    gz = np.zeros(shape)
    xy = []
    xy.append(x)
    xy.append(y)
    gpad, nps = gridder.pad_array(gz)
    xyp = gridder.pad_coords(xy, gz.shape, nps)
    [Yp, Xp] = np.meshgrid(xyp[1], xyp[0])
    assert_equal(Yp.shape, gpad.shape)
    assert_almost(Yp[nps[0][0]:-nps[0][1], nps[1][0]:-nps[1][1]].ravel(), y)
    assert_almost(Xp[nps[0][0]:-nps[0][1], nps[1][0]:-nps[1][1]].ravel(), x)


def test_fails_if_npd_incorrect_dimension():
    'gridder.pad_array raises error if given improper dimension on npadding'
    s = (101, 172)
    x, y, z = gridder.regular((-5000., 5000., -5000., 5000.), s, z=-150)
    g = z.reshape(s)
    npdt = 128
    assert_raises(ValueError, gridder.pad_array, g, npd=npdt)
    npdt = (128, 256, 142)
    assert_raises(ValueError, gridder.pad_array, g, npd=npdt)
    prng = RandomState(12345)
    g = prng.rand(50)
    assert_raises(ValueError, gridder.pad_array, g, npd=npdt)


def test_fails_if_npd_lessthan_arraydim():
    'gridder.pad_array raises error if given npad is less than array length'
    shape = (101, 172)
    x, y, z = gridder.regular((-5000., 5000., -5000., 5000.), shape, z=-150)
    g = z.reshape(shape)
    npdt = (128, 128)
    assert_raises(ValueError, gridder.pad_array, g, npd=npdt)
    prng = RandomState(12345)
    g = prng.rand(20)
    npdt = 16
    assert_raises(ValueError, gridder.pad_array, g, npd=npdt)
