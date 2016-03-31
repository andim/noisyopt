Tutorial - Minimize a noisy function
====================================

The following tutorial shows how to find a local minimum of a
simple function using the `noisyopt` library.

The problem is to solve the following optimization problem

.. math::

    \min_{\boldsymbol x \in \Omega} f

    f(x_1, x_2) = x_1^2 + x_2^2

    \Omega = [-3.0, 3.0] \times [0.5, 5.0],

where we do not have access to the function f directly, but only
to some noisy approximation

.. math::

    \tilde f = f + \xi, \quad \xi \sim \mathcal{N}(0, 0.1^2).
    
First we need to import the ``minimize`` function from the `noisyopt` package::

  >>> import numpy as np
  >>> from noisyopt import minimize

Then we need to define the objective of the function::

  >>> def obj(x):
    ...     return (x**2).sum() + np.random.randn(1)

We now define the domain of the problem using bound constraints::

  >>> bounds=[[-3.0, 3.0], [0.5, 5.0]]

And we define the initial guess::

  >>> x0 = np.array([-2.0, 2.0])
               
The algoritm is called using the ``minimize`` function. The `minimize`
functions accepts the problem objective ``obj`` and block constraints::

  >>> res = minimize(obj, bounds=bounds, x0=x0, deltatol=0.1, errorcontrol=True)

In the above we use the default settings of the `DIRECT` algorithm.
It us possible to costumize the algorithm using the parameters of
the ``minimize`` function (see :py:func:`scipydirect.minimize`).

The ``minimize`` function returns a result object ``res`` making accessible among 
other the optimal point, ``res.x``, and the value of the objective at the
optimum, ``res.fun``::

  >>> print res
       funse: 0.0089061854063998726
     success: True
        free: array([False, False], dtype=bool)
        nfev: 1320
         fun: 0.23298649773605173
           x: array([-0.05,  0.5 ])
     message: 'convergence within deltatol'
         nit: 10
