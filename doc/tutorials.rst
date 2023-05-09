.. include:: bibliography.txt

.. _tutorials:

Tutorials
=========

The following tutorials show how to use iminuit and explore different aspects of the library. The order is the recommended reading order, the later entries are about more and more specialized applications.
Important for most users are only the first two entries.

.. toctree::
    :maxdepth: 1

    notebooks/basic
    notebooks/cost_functions
    notebooks/error_bands
    notebooks/interactive
    notebooks/simultaneous_fits
    notebooks/template_fits
    notebooks/template_model_mix
    notebooks/conditional_variable
    notebooks/scipy_and_constraints
    notebooks/roofit
    notebooks/external_minimizer
    notebooks/generic_least_squares
    notebooks/cython

RooFit tutorials
----------------

The following tutorials correspond to a `RooFit`_ tutorial, which performs the same task but uses only iminuit and other libraries from the standard Python scientific stack and Scikit-HEP. It can be used as a learning resource or to decide which toolset to use.

The list is incomplete. If you would like to see more tutorials, please leave an issue on Github with your request or feel free to make a PR to provide it yourself.

.. toctree::
    :maxdepth: 1

    notebooks/roofit/rf101_basics
    notebooks/roofit/rf109_chi2residpull
