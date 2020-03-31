.. include:: references.txt

.. _tutorials:

Tutorials
=========

All the tutorials are in tutorial directory. You can view them online too:

`Basic tutorial <http://nbviewer.ipython.org/urls/raw.github.com/scikit-hep/iminuit/master/tutorial/basic_tutorial.ipynb>`_
---------------------------------------------------------------------------------------------------------------------------------

Covers the basics of using iminuit.

`iminuit and automatic differentiation with JAX <http://nbviewer.ipython.org/urls/raw.github.com/scikit-hep/iminuit/master/tutorial/automatic_differentiation.ipynb>`_
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

How to compute function gradients for iminuit with jax_ and accelerate Python code with JAX's JIT compiler. Spoiler: a **32x** speed up over plain numpy is achieved. Also discusses how to do a least-squares fit with data that has uncertainties in *x* and *y*.

Outdated Cython tutorials
-------------------------

The following two tutorials are outdated. Users who want to speed up their fits should try the just-in-time compilers provided by numba_ or jax_ in CPython or use iminuit in PyPy to accelerate the computation. This is much simpler than using Cython and may achieve even better performance.

- `Advanced tutorial <http://nbviewer.ipython.org/urls/raw.github.com/scikit-hep/iminuit/master/tutorial/advanced_tutorial.ipynb>`_.
  Shows how to speed up the computation of the cost function with Cython.

- `Hard Core Cython tutorial <http://nbviewer.ipython.org/urls/raw.github.com/scikit-hep/iminuit/master/tutorial/hard_core_tutorial.ipynb>`_.
  Goes into more detail on how to use Cython.
