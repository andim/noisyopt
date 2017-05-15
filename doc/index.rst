.. noisyopt documentation master file

Welcome to noisyopt's documentation!
====================================

`noisyopt`  is concerned with solving an (possibly bound-constrained) optimization problem of the kind

.. math::

    \min_{\boldsymbol x} f(\boldsymbol x) = \min_{\boldsymbol x} \mathbb{E}[F(\boldsymbol x, \xi)]

where evaluations of the function f are not directly possible, but only evaluation
of the function F. The expectation value of F is f, but F also depends on some random
noise.
Such optimization problems are known under various names, such as stochastic approximation, stochastic optimization/programming, or noisy optimization.
They arise in various contexts from `Simulation-based optimization <https://en.wikipedia.org/wiki/Simulation-based_optimization>`_, to `machine learning <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_.

To solve such optimization problems the package currently implements two (derivative-free) algorithms:

- a robust pattern search algorithm with an adaptive number of function evaluations [Mayer2016]_
- a stochastic approximation algorithm, namely `simultaneous perturbation stochastic approximation <http://www.jhuapl.edu/SPSA/>`_ [Spall1998]_

Further algorithms might be added in the future -- you are invited to `contribute <https://github.com/andim/noisyopt/blob/master/CONTRIBUTING.md>`_!
The package also contains a function to find the root of a noisy function by a bisection algorithm with an adaptive number of function evaluations.

Noisyopt is concerned with local optimization, if you are interested in global optimization you might want to have a look at Bayesian optimization techniques (see e.g. `scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`_).
For optimizing functions that are not noisy take a look at `scipy.optimize <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_.

Documentation
-------------

You can install `noisyopt` using `pip`::

    pip install noisyopt

Minimal usage example::

    import numpy as np
    from noisyopt import minimizeCompass

    def obj(x):
        return (x**2).sum() + 0.1*np.random.randn()

    res = minimizeCompass(obj, x0=[1.0, 2.0], deltatol=0.1, paired=False)

For further documentation see below or the `usage examples <https://github.com/andim/noisyopt/tree/master/examples>`_.

.. toctree::
   :maxdepth: 1

   tutorial
   api
   changelog

Further reading
---------------

If you use and like this package, you might want to read and cite the papers describing the implemented algorithms:

.. [Mayer2016] Mayer, A.; Mora, T.; Rivoire, O. & Walczak, A. M. Diversity of immune strategies explained by adaptation to pathogen statistics. PNAS, 2016, 113(31), 8630-8635. Relevant section is in the Supplementary Information entitled "Pattern-search based optimization for problems with noisy function evaluations".
.. [Spall1998] Spall, JC. Implementation of the simultaneous perturbation algorithm for stochastic optimization. Aerospace and Electronic Systems, IEEE Transactions on, IEEE, 1998, 34, 817-823 
