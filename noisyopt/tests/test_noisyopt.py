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

    ## simple tests
    root = noisyopt.bisect(lambda x: x, -2, 2, xtol=xtol)
    npt.assert_allclose(root, 0.0, atol=xtol)

    root = noisyopt.bisect(lambda x: x-1, -2, 2, xtol=xtol)
    npt.assert_allclose(root, 1.0, atol=xtol)

    ## extrapolate if 0 outside of interval
    root = noisyopt.bisect(lambda x: x, 1, 2, xtol=xtol)
    npt.assert_allclose(root, 0.0, atol=xtol)
    npt.assert_raises(noisyopt.BisectException,
                      noisyopt.bisect, lambda x: x, 1, 2,
                      xtol=xtol, outside='raise')
    
    ## extrapolate with nonlinear function
    root = noisyopt.bisect(lambda x: x+x**2, 1.0, 2, xtol=xtol)
    assert root < 1.0

    ## test with stochastic function
    xtol = 1e-1
    func = lambda x: x - 0.25 + np.random.normal(scale=0.01)
    root = noisyopt.bisect(noisyopt.AveragedFunction(func), -2, 2, xtol=xtol,
                           errorcontrol=True)
    npt.assert_allclose(root, 0.25, atol=xtol)

def test_AveragedFunction():
    ## averaging a simple function 
    func = lambda x: np.asarray(x).sum()
    avfunc = noisyopt.AveragedFunction(func, N=30)
    av, avse = avfunc([1.0, 1.0])
    npt.assert_equal(av, 2.0)
    npt.assert_equal(avse, 0.0) 

    # se of function value difference between two points is zero
    # (as function evaluation is not stochastic)
    diffse = avfunc.diffse([1.0, 1.0], [2.0, 1.0])
    npt.assert_equal(diffse, 0.0)

    ## changing the number of evaluations
    avfunc.N *= 2
    npt.assert_equal(avfunc.N, 60)

    ## averaging a stochastic function
    func = lambda x: np.asarray(x).sum() + np.random.randn()
    avfunc = noisyopt.AveragedFunction(func, N=30)
    # check that reevaluation gives the same thing due to caching
    av30_1, avse30_1 = avfunc([1.0, 1.0])
    av30_2, avse30_2 = avfunc([1.0, 1.0])
    npt.assert_equal(av30_1, av30_2)
    npt.assert_equal(avse30_1, avse30_2)
    # check that se decreases if 
    avfunc.N *= 2
    av60, avse60 = avfunc([1.0, 1.0])
    assert av30_1 != av60
    assert avse30_1 > avse60

    # test with floating point N
    noisyopt.AveragedFunction(func, N=30.0, paired=True)

if __name__ == '__main__':
    npt.run_module_suite()
