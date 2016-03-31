.. noisyopt documentation master file

Welcome to noisyopt's documentation!
====================================

`noisyopt`  is concerned with solving an (possibly bound-constrained) optimization problem of the kind

.. math::

    \min_{ l_i < x_i < u_i} f(\boldsymbol x),

where the evaluations do not yield the precise function value, but the value plus some noise.
The package implements a pattern search algorithm with an adaptive number of function
evaluations to handle the stochasticity in function evaluations.

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
