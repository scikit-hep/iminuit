.. |iminuit| image:: doc/_static/iminuit_logo.svg
   :alt: iminuit

|iminuit|
=========

.. version-marker-do-not-remove

.. image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org
.. image:: https://img.shields.io/pypi/v/iminuit.svg
   :target: https://pypi.org/project/iminuit
.. image:: https://img.shields.io/conda/vn/conda-forge/iminuit.svg
   :target: https://github.com/conda-forge/iminuit-feedstock
.. image:: https://coveralls.io/repos/github/scikit-hep/iminuit/badge.svg?branch=develop
   :target: https://coveralls.io/github/scikit-hep/iminuit?branch=develop
.. image:: https://readthedocs.org/projects/iminuit/badge/?version=latest
   :target: https://iminuit.readthedocs.io/en/stable
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3949207.svg
   :target: https://doi.org/10.5281/zenodo.3949207
.. image:: https://img.shields.io/badge/ascl-2108.024-blue.svg?colorB=262255
   :target: https://ascl.net/2108.024
   :alt: ascl:2108.024
.. image:: https://img.shields.io/gitter/room/Scikit-HEP/iminuit
   :target: https://gitter.im/Scikit-HEP/iminuit
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/iminuit/develop?filepath=doc%2Ftutorial

``iminuit`` is a Jupyter-friendly Python interface for the ``Minuit2`` C++ library maintained by CERN's ROOT team.

Minuit was designed to minimise statistical cost functions, for likelihood and least-squares fits of parametric models to data. It provides the best-fit parameters and error estimates from likelihood profile analysis.

- Supported CPython versions: 3.6+
- Supported PyPy versions: 3.6+
- Supported platforms: Linux, OSX and Windows.

The iminuit package comes with additional features:

- Builtin cost functions for statistical fits

  - Binned and unbinned maximum-likelihood
  - Template fits with error propagation [H. Dembinski, A. Abldemotteleb, Eur.Phys.J.C 82 (2022) 11, 1043](https://doi.org/10.1140/epjc/s10052-022-11019-z)
  - Non-linear regression with (optionally robust) weighted least-squares
  - Gaussian penalty terms
  - Cost functions can be combined by adding them: ``total_cost = cost_1 + cost_2``
- Support for SciPy minimisers as alternatives to Minuit's Migrad algorithm (optional)
- Support for Numba accelerated functions (optional)

Dependencies
------------

``iminuit`` is will always be a lean package which only depends on ``numpy``, but additional features are enabled if the following optional packages are installed
- ``matplotlib``: Visualization of fitted model for builtin cost functions
- ``ipywidgets``: Interactive fitting (also requires ``matplotlib``)
- ``scipy``: Compute Minos intervals for arbitrary confidence levels
- ``unicodeitplus``: Render names of model parameters in simple LaTeX as Unicode

Documentation
-------------

Checkout our large and comprehensive list of `tutorials`_ that take you all the way from beginner to power user. For help and how-to questions, please use the `discussions`_ on GitHub or `gitter`_.

**Lecture by Glen Cowan**

`In the exercises to his lecture for the KMISchool 2022 <https://github.com/KMISchool2022>`_, Glen Cowan shows how to solve statistical problems in Python with iminuit. You can find the lectures and exercises on the Github page, which covers both frequentist and Bayesian methods.

`Glen Cowan <https://scholar.google.com/citations?hl=en&user=ljQwt8QAAAAJ&view_op=list_works>`_ is a known for his papers and international lectures on statistics in particle physics, as a member of the Particle Data Group, and as author of the popular book `Statistical Data Analysis <https://www.pp.rhul.ac.uk/~cowan/sda/>`_.

In a nutshell
-------------

iminuit is intended to be used with a user-provided negative log-likelihood function or least-squares function. Standard functions are included in ``iminuit.cost``, so you don't have to write them yourself. The following example shows how iminuit is used with a dummy least-squares function.

.. code-block:: python

    from iminuit import Minuit

    def cost_function(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(cost_function, x=0, y=0, z=0)

    m.migrad()  # run optimiser
    m.hesse()   # run covariance estimator

.. raw::html

   <table>
      <tr>
         <th colspan="5" style="text-align:center" title="Minimizer"> Migrad </th>
      </tr>
      <tr>
         <td colspan="2" style="text-align:left" title="Minimum value of function"> FCN = 6.731e-18 </td>
         <td colspan="3" style="text-align:center" title="Total number of function and (optional) gradient evaluations"> Nfcn = 52 </td>
      </tr>
      <tr>
         <td colspan="2" style="text-align:left" title="Estimated distance to minimum and goal"> EDM = 6.73e-18 (Goal: 0.0002) </td>
         <td colspan="3" style="text-align:center" title="Total run time of algorithms">  </td>
      </tr>
      <tr>
         <td colspan="2" style="text-align:center;background-color:#92CCA6;color:black"> Valid Minimum </td>
         <td colspan="3" style="text-align:center;background-color:#92CCA6;color:black"> No Parameters at limit </td>
      </tr>
      <tr>
         <td colspan="2" style="text-align:center;background-color:#92CCA6;color:black"> Below EDM threshold (goal x 10) </td>
         <td colspan="3" style="text-align:center;background-color:#92CCA6;color:black"> Below call limit </td>
      </tr>
      <tr>
         <td style="text-align:center;background-color:#92CCA6;color:black"> Covariance </td>
         <td style="text-align:center;background-color:#92CCA6;color:black"> Hesse ok </td>
         <td style="text-align:center;background-color:#92CCA6;color:black" title="Is covariance matrix accurate?"> Accurate </td>
         <td style="text-align:center;background-color:#92CCA6;color:black" title="Is covariance matrix positive definite?"> Pos. def. </td>
         <td style="text-align:center;background-color:#92CCA6;color:black" title="Was positive definiteness enforced by Minuit?"> Not forced </td>
      </tr>
   </table><table>
      <tr>
         <td></td>
         <th title="Variable name"> Name </th>
         <th title="Value of parameter"> Value </th>
         <th title="Hesse error"> Hesse Error </th>
         <th title="Minos lower error"> Minos Error- </th>
         <th title="Minos upper error"> Minos Error+ </th>
         <th title="Lower limit of the parameter"> Limit- </th>
         <th title="Upper limit of the parameter"> Limit+ </th>
         <th title="Is the parameter fixed in the fit"> Fixed </th>
      </tr>
      <tr>
         <th> 0 </th>
         <td> x </td>
         <td> 2 </td>
         <td> 1 </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
      </tr>
      <tr>
         <th> 1 </th>
         <td> y </td>
         <td> 3 </td>
         <td> 1 </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
      </tr>
      <tr>
         <th> 2 </th>
         <td> z </td>
         <td> 4 </td>
         <td> 1 </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
         <td>  </td>
      </tr>
   </table><table>
      <tr>
         <td></td>
         <th> x </th>
         <th> y </th>
         <th> z </th>
      </tr>
      <tr>
         <th> x </th>
         <td> 1 </td>
         <td style="background-color:rgb(250,250,250);color:black"> -0 </td>
         <td style="background-color:rgb(250,250,250);color:black"> -0 </td>
      </tr>
      <tr>
         <th> y </th>
         <td style="background-color:rgb(250,250,250);color:black"> -0 </td>
         <td> 1 </td>
         <td style="background-color:rgb(250,250,250);color:black"> -0 </td>
      </tr>
      <tr>
         <th> z </th>
         <td style="background-color:rgb(250,250,250);color:black"> -0 </td>
         <td style="background-color:rgb(250,250,250);color:black"> -0 </td>
         <td> 1 </td>
      </tr>
   </table>

Interactive fitting
-------------------

iminuit optionally supports an interactive fitting mode in Jupyter notebooks.

.. image:: doc/_static/interactive_demo.gif
   :alt: Animated demo of an interactive fit in a Jupyter notebook

Partner projects
----------------

* `boost-histogram` from Scikit-HEP provides fast generalized histograms that you can use with the builtin cost functions.
* `numba_stats`_ provides faster implementations of probability density functions than scipy, and a few specific ones used in particle physics that are not in scipy.
* `jacobi`_ provides a robust, fast, and accurate calculation of the Jacobi matrix of any transformation function and building a function for generic error propagation.

Versions
--------

**The current 2.x series has introduced breaking interfaces changes with respect to the 1.x series.**

All interface changes are documented in the `changelog`_ with recommendations how to upgrade. To keep existing scripts running, pin your major iminuit version to <2, i.e. ``pip install 'iminuit<2'`` installs the 1.x series.

.. _changelog: https://iminuit.readthedocs.io/en/stable/changelog.html
.. _tutorials: https://iminuit.readthedocs.io/en/stable/tutorials.html
.. _discussions: https://github.com/scikit-hep/iminuit/discussions
.. _gitter: https://gitter.im/Scikit-HEP/iminuit
.. _jacobi: https://github.com/hdembinski/jacobi
.. _numba_stats: https://github.com/HDembinski/numba-stats
