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
to some noisy approximation.

.. math::

    \tilde f = f + \xi, \quad \xi \sim \mathcal{N}(0, 0.1^2).

This is obviously a toy example (the solution (0.0, 0.5) is obvious from
inspection of the problem), but similar problems arise in practice.
    
First we need to import the ``minimizeCompass`` function from the `noisyopt` package::

  >>> import numpy as np
  >>> from noisyopt import minimizeCompass

Then we need to define the objective of the function::

  >>> def obj(x):
    ...     return (x**2).sum() + 0.1*np.random.randn()

We now define the domain of the problem using bound constraints::

  >>> bounds = [[-3.0, 3.0], [0.5, 5.0]]

And we define the initial guess::

  >>> x0 = np.array([-2.0, 2.0])
               
The pattern search based optimization algorithm is called using the ``minimizeCompass`` function. The `minimizeCompass` functions accepts the problem objective ``obj`` and bound constraints::

  >>> res = minimizeCompass(obj, bounds=bounds, x0=x0, deltatol=0.1, paired=False)

It is possible to further customize the algorithm using the parameters of
the ``minimizeCompass`` function (see :py:func:`noisyopt.minimizeCompass`). Alternatively we can also try a different algorithm implementing a simultaneous perturbation stochastic approximation algorithm (see :py:func:`noisyopt.minimizeSPSA`).

The ``minimizeCompass`` function returns a result object ``res`` making accessible among 
other the optimal point, ``res.x``, and the value of the objective at the
optimum, ``res.fun``::

  >>> print(res)
      free: array([False, False], dtype=bool)
       fun: 0.25068227287617101
     funse: 0.0022450327079771111
   message: 'convergence within deltatol'
      nfev: 9510
       nit: 9
   success: True
         x: array([ 0. ,  0.5])

As instructed, the algorithm finds the correct solution respecting the bounds (the unconstrained optimum would be at [0, 0]).

Alternatively we can use the SPSA algorithm also included in the library to get an equivalent result (up to the finite accuracy of the optimization algorithm):: 

  >>> from noisyopt import minimizeSPSA
  >>> res = minimizeSPSA(obj, bounds=bounds, x0=x0, niter=1000, paired=False)
  >>> print(res)
       fun: 0.40851628698699205
   message: 'terminated after reaching max number of iterations'
      nfev: 2000
       nit: 1000
   success: True
         x: array([ 0.0359265,  0.5      ])

  
Further examples
----------------

A usage example on a real-world problem can be found at http://github.com/andim/evolimmune/ including using the :py:func:`noisyopt.bysect` `routine <https://github.com/andim/evolimmune/blob/master/fig2/run_phases.py>`_.

