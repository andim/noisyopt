[![License](https://img.shields.io/pypi/l/noisyopt.svg)](https://github.com/andim/noisyopt/blob/master/LICENSE)
![Status](https://img.shields.io/pypi/status/noisyopt.svg)
[![Latest release](https://img.shields.io/pypi/v/noisyopt.svg)](https://pypi.python.org/pypi/noisyopt)
[![Build Status](https://travis-ci.org/andim/noisyopt.svg?branch=master)](https://travis-ci.org/andim/noisyopt)
[![Documentation Status](https://readthedocs.org/projects/noisyopt/badge/?version=latest)](https://noisyopt.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/andim/noisyopt/badge.svg?branch=master)](https://coveralls.io/github/andim/noisyopt?branch=master)

# noisyopt

Python library for optimization of noisy functions.

Currently variants of the following algorithms are implemented:
- pattern search with adaptive sampling
- simultaneous perturbation stochastic approximation

Optionally bound constraints on variables are possible.

## Installation

Noisyopt is on [PyPI](https://pypi.python.org/pypi/noisyopt/) so you can install it using `pip install noisyopt`.

Alternatively you can install it from source by obtaining the source code from [Github](https://github.com/andim/noisyopt) and then running `python setup.py install` in the main directory. If you install from source, you first need to install `numpy` and `scipy` if these packages are not already installed.

## Documentation

You can access the documentation online at [Read the docs](http://noisyopt.readthedocs.io/en/latest/). If you install from source you can generate a local version by running `make html` from the `doc` directory.

## Testing

You can run the testsuite using the py.test testing framework `py.test`.

## Support and contributing

For bug reports and enhancement requests use the [Github issue tool](http://github.com/andim/noisyopt/issues/new), or (even better!) open a [pull request](http://github.com/andim/noisyopt/pulls) with relevant changes. If you have any questions don't hesitate to contact me by email (andisspam@gmail.com) or Twitter ([@andisspam](http://twitter.com/andisspam)).

You are cordially invited to [contribute](https://github.com/andim/noisyopt/blob/master/CONTRIBUTING.md) to the further development of noisyopt!
