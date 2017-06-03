[![License](https://img.shields.io/pypi/l/noisyopt.svg)](https://github.com/andim/noisyopt/blob/master/LICENSE)
[![Latest release](https://img.shields.io/pypi/v/noisyopt.svg)](https://pypi.python.org/pypi/noisyopt)
[![Py2.7/3.x](https://img.shields.io/pypi/pyversions/noisyopt.svg)](https://pypi.python.org/pypi/noisyopt)

![Status](https://img.shields.io/pypi/status/noisyopt.svg)
[![Build Status](https://travis-ci.org/andim/noisyopt.svg?branch=master)](https://travis-ci.org/andim/noisyopt)
[![Documentation Status](https://readthedocs.org/projects/noisyopt/badge/?version=latest)](https://noisyopt.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/andim/noisyopt/badge.svg?branch=master)](https://coveralls.io/github/andim/noisyopt?branch=master)

[![JOSS](http://joss.theoj.org/papers/4d17c8d6e2cfe6505ca5ccdace5e123b/status.svg)](http://joss.theoj.org/papers/4d17c8d6e2cfe6505ca5ccdace5e123b)
[![DOI](https://zenodo.org/badge/54976198.svg)](https://zenodo.org/badge/latestdoi/54976198)


# Noisyopt: A python library for optimizing noisy functions

Currently the following algorithms are implemented:
- robust pattern search with adaptive sampling
- simultaneous perturbation stochastic approximation
Both algorithms support bound constraints and do not require to explicitely calculate the gradient of the function.

We do not attempt to find global optima -- look at [`scikit-optimize`](https://github.com/scikit-optimize/scikit-optimize) for Bayesian optimization algorithms aimed at finding global optima to noisy optimization problems.
For optimizing functions that are not noisy take a look at [`scipy.optimize`](http://docs.scipy.org/doc/scipy/reference/optimize.html).

## Installation

Noisyopt is on [PyPI](https://pypi.python.org/pypi/noisyopt/) so you can install it using `pip install noisyopt`.

Alternatively you can install it from source by obtaining the source code from [Github](https://github.com/andim/noisyopt) and then running `python setup.py install` in the main directory. If you install from source, you first need to install `numpy` and `scipy` if these packages are not already installed.

## Getting started

Find the minimum of the noisy function `obj(x)` with `noisyopt`:

```python
import numpy as np
from noisyopt import minimizeCompass

def obj(x):
    return (x**2).sum() + 0.1*np.random.randn()

bounds = [[-3.0, 3.0], [0.5, 5.0]]
x0 = np.array([-2.0, 2.0])
res = minimizeCompass(obj, bounds=bounds, x0=x0, deltatol=0.1, paired=False)
```

## Documentation

You can access the documentation online at [Read the docs](http://noisyopt.readthedocs.io/en/latest/). If you install from source you can generate a local version by running `make html` from the `doc` directory.

## Support and contributing

For bug reports and enhancement requests use the [Github issue tool](http://github.com/andim/noisyopt/issues/new), or (even better!) open a [pull request](http://github.com/andim/noisyopt/pulls) with relevant changes. If you have any questions don't hesitate to contact me by email (andimscience@gmail.com) or Twitter ([@andimscience](http://twitter.com/andimscience)).

You can run the testsuite by running `pytest` in the top-level directory.

You are cordially invited to [contribute](https://github.com/andim/noisyopt/blob/master/CONTRIBUTING.md) to the further development of noisyopt!
