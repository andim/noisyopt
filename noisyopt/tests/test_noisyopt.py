import numpy as np
import numpy.testing as npt
import noisyopt

def test_minimize():
    def quadratic(x):
        return (x**2).sum()
    res = noisyopt.minimize(quadratic, np.asarray([0.5, 1.0]))
    npt.assert_equal(res.x, [0.0, 0.0])
