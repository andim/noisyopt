# This is a minimal usage example
# see also http://noisyopt.readthedocs.io/en/latest/tutorial.html

import numpy as np
from noisyopt import minimizeCompass, minimizeSPSA

def obj(x):
    return (x**2).sum() + 0.1*np.random.randn()

bounds = [[-3.0, 3.0], [0.5, 5.0]]
x0 = np.array([-2.0, 2.0])
res = minimizeCompass(obj, bounds=bounds, x0=x0, deltatol=0.1, paired=False)
print(res)

res = minimizeSPSA(obj, bounds=bounds, x0=x0, niter=1000, paired=False)
print(res)
