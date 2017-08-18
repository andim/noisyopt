# This is a minimal usage example for the root finding algorithm
import numpy as np
from noisyopt import bisect, AveragedFunction

# definition of noisy function of which root should be found
def func(x):
    return x + 0.1*np.random.randn()

avfunc = AveragedFunction(func)
root = bisect(avfunc, -2, 2)
print(root)
