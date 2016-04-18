.. noisyopt documentation master file

Welcome to noisyopt's documentation!
====================================

`noisyopt`  is concerned with solving an (possibly bound-constrained) optimization problem of the kind

.. math::

    \min_{ l_i < x_i < u_i} f(\boldsymbol x) = \min_{l_i < x_i < u_i} \mathbb{E}[F(\boldsymbol x, \xi)]

where evaluations of the function f are not directly possible, but only evaluation
of the function F. The expectation value of F is f, but F also depends on some random
noise. 
Such optimization problems arise e.g. in the context of `Simulation-based optimization
<https://en.wikipedia.org/wiki/Simulation-based_optimization>`_.
To solve this optimization problems the package implements a pattern search
algorithm with an adaptive number of function evaluations to handle the stochasticity
in function evaluations.

The package also contains a function to find the root of a noisy function by a bisection
algorithm with an adaptive number of function evaluations.

Documentation
-------------

To see how to install it, please refer to the `README file 
<https://github.com/andim/noisyopt/blob/master/README.md>`_ in the Github repository.

.. toctree::
   :maxdepth: 2

   tutorial
   reference/index

Further reading
---------------

The algorithm is described in the Supplementary Information of [Mayer2016]_.

.. [Mayer2016] Mayer, A.; Mora, T.; Rivoire, O. & Walczak, A. M. Diversity of immune strategies explained by adaptation to pathogen statistics, bioRxiv, 2015
