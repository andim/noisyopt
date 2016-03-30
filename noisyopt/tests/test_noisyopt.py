import numpy as np
import numpy.testing as npt
import noisyopt

def test_minimize():

    deltatol = 1e-3
    ## basic testing without stochasticity
    def quadratic(x):
        return (x**2).sum()

    res = noisyopt.minimize(quadratic, np.asarray([0.5, 1.0]), deltatol=deltatol)
    npt.assert_allclose(res.x, [0.0, 0.0], atol=deltatol)
    npt.assert_equal(res.free, [False, False])

    res = noisyopt.minimize(quadratic, np.asarray([2.5, -3.2]), deltatol=deltatol)
    npt.assert_allclose(res.x, [0.0, 0.0], atol=deltatol)
    npt.assert_equal(res.free, [False, False])

    res = noisyopt.minimize(quadratic, np.asarray([2.5, -3.2, 0.9, 10.0, -0.3]),
                            deltatol=deltatol)
    npt.assert_allclose(res.x, np.zeros(5), atol=deltatol)
    npt.assert_equal(res.free, [False, False, False, False, False])

    ## test bound handling
    res = noisyopt.minimize(quadratic, np.asarray([0.5, 0.5]),
                            bounds=np.asarray([[0, 1], [0, 1]]), deltatol=deltatol)
    npt.assert_allclose(res.x, [0.0, 0.0], atol=deltatol)
    npt.assert_equal(res.free, [False, False])

    res = noisyopt.minimize(quadratic, np.asarray([0.8, 0.8]),
                            bounds=np.asarray([[0.5, 1], [0.5, 1]]),
                            deltatol=deltatol)
    npt.assert_allclose(res.x, [0.5, 0.5], atol=deltatol)
    npt.assert_equal(res.free, [False, False])


    ## test determination of unconstrained variables
    def quadratic_except_last(x):
        return (x[:-1]**2).sum()

    res = noisyopt.minimize(quadratic_except_last, np.asarray([0.5, 1.0]))
    npt.assert_approx_equal(res.x[0], 0.0)
    npt.assert_equal(res.free, [False, True])

    ## test errorcontrol for stochastic function
    def stochastic_quadratic(x, seed=None):
        prng = np.random if seed is None else np.random.RandomState(seed)
        return (x**2).sum() + prng.randn(1) + 0.5*np.random.randn(1)

    deltatol = 0.5
    # test unpaired
    res = noisyopt.minimize(stochastic_quadratic, np.array([4.55, 3.0]),
                            deltainit=2.0, deltatol=deltatol,
                            errorcontrol=True)
    npt.assert_allclose(res.x, [0.0, 0.0], atol=deltatol)
    npt.assert_equal(res.free, [False, False])
    # test paired
    res = noisyopt.minimize(stochastic_quadratic, np.array([4.55, 3.0]),
                            deltainit=2.0, deltatol=deltatol,
                            errorcontrol=True, paired=True)
    npt.assert_allclose(res.x, [0.0, 0.0], atol=deltatol)
    npt.assert_equal(res.free, [False, False])

def test_bisect():

    xtol = 1e-6 
    significant = 6
    ## simple tests
    root = noisyopt.bisect(lambda x: x, -2, 2, xtol=xtol)
    npt.assert_approx_equal(root, 0.0, significant=significant)

    root = noisyopt.bisect(lambda x: x-1, -2, 2, xtol=xtol)
    npt.assert_approx_equal(root, 1.0, significant=significant)

    ## extrapolate if 0 outside of interval
    root = noisyopt.bisect(lambda x: x, 1, 2, xtol=xtol)
    npt.assert_approx_equal(root, 0.0, significant=significant)
