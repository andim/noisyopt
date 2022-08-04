.. api reference

API Reference
=============

.. toctree::
   :maxdepth: 2

.. currentmodule:: noisyopt

minimizeCompass
~~~~~~~~~~~~~~~

.. autofunction:: minimizeCompass

minimizeSPSA
~~~~~~~~~~~~

.. autofunction:: minimizeSPSA

minimize
~~~~~~~~

.. autofunction:: minimize

bisect
~~~~~~

.. autofunction:: bisect


Average classes
~~~~~~~~~~~~~~~

These helper classes perform averages over function values. They provide extra logic such as tests whether function values differ signficantly.

.. autoclass:: AverageBase
    :members:
.. autoclass:: AveragedFunction
    :members:
    :inherited-members:
.. autoclass:: DifferenceFunction
    :members:
    :inherited-members:
