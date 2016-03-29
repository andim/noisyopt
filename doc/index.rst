.. noisyopt documentation master file

Welcome to noisyopt's documentation!
====================================

`noisyopt`  is concerned with solving an (possibly bound-constrained) optimization problem of the kind

.. math::

    \min_{ l_i < x_i < u_i} f(\boldsymbol x),

where the evaluations do not yield the precise function value, but the value plus some noise [Mayer2016]_.

To see how to use it, please refer to the `README file 
<https://github.com/andim/noisyopt/blob/master/README.md>`_ in the Github repository.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   reference/index

Further reading
---------------

.. [Mayer2016] Mayer, A.; Mora, T.; Rivoire, O. & Walczak, A. M. Diversity of immune strategies explained by adaptation to pathogen statistics, bioRxiv, 2015
