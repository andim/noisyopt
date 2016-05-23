.. noisyopt documentation master file

Welcome to noisyopt's documentation!
====================================

`noisyopt`  is concerned with solving an (possibly bound-constrained) optimization problem of the kind

.. math::

    \min_{\boldsymbol x} f(\boldsymbol x) = \min_{\boldsymbol x} \mathbb{E}[F(\boldsymbol x, \xi)]

where evaluations of the function f are not directly possible, but only evaluation
of the function F. The expectation value of F is f, but F also depends on some random
noise.
Such optimization problems are known under names such as stochastic optimization/programming, or noisy optimization.
They arise e.g. in the context of `Simulation-based optimization <https://en.wikipedia.org/wiki/Simulation-based_optimization>`_.

To solve this optimization problems the package implements two algorithms:

- a novel pattern search algorithm with an adaptive number of evaluations of F to control the noise in the approximation of f [Mayer2016]_
- a stochastic gradient-descent like algorithm, called `simultaneous perturbation stochastic approximation <http://www.jhuapl.edu/SPSA/>`_ [Spall1998]_

The package also contains a function to find the root of a noisy function by a bisection
algorithm with an adaptive number of function evaluations.

Documentation
-------------

To see how to install the package, please refer to the `README file 
<https://github.com/andim/noisyopt/blob/master/README.md>`_ in the Github repository.

.. toctree::
   :maxdepth: 1

   tutorial
   reference/index
   changelog

Further reading
---------------

If you use this package, you might want to read and cite the papers describing the implemented algorithms:

.. [Mayer2016] Mayer, A.; Mora, T.; Rivoire, O. & Walczak, A. M. Diversity of immune strategies explained by adaptation to pathogen statistics. bioRxiv, 2015. Relevant information is in a section of the Supplementary Information entitled "Pattern-search based optimization for problems with noisy function evaluations".
.. [Spall1998] Spall, JC. Implementation of the simultaneous perturbation algorithm for stochastic optimization. Aerospace and Electronic Systems, IEEE Transactions on, IEEE, 1998, 34, 817-823 
