.. include:: references.txt

.. _changelog:

Changelog
=========

1.3.10 (March, 31, 2020)
------------------------

Bug-fixes
~~~~~~~~~
- sdist package was broken, this was fixed by @henryiii

Implementation
~~~~~~~~~~~~~~
- Allow HESSE to be called without running MIGRAD first

Documentation
~~~~~~~~~~~~~
- Added tutorial to show how iminuit can compute parameter errors for other minimizers

Other
~~~~~
- @henryiii added a CI test to check the sdist package and the MANIFEST

1.3.9 (March, 31, 2020)
-----------------------

Bug-fixes
~~~~~~~~~
- `draw_contour` now accepts an integer for `bound` keyword as advertised in the docs
- fixed wrong EDM goal in iminuit reports, was off by factor 5 in some

Interface
~~~~~~~~~
- removed the undocumented keyword "args" in `(draw_)contour`, `(draw_)profile`
- removed misleading "show_sigma" keyword in `draw_contour`
- deprecated `Minuit.is_fixed`, replaced by `.fixed` attribute
- deprecated `Minuit.set_strategy`, assign to `Minuit.strategy` instead
- deprecated `Minuit.set_errordef`, assign to `Minuit.errordef` instead
- deprecated `Minuit.set_print_level`, assign to `Minuit.print_level` instead
- deprecated `Minuit.print_fmin`, `Minuit.print_matrix`, `Minuit.print_param`, `Minuit.print_initial_param`, `Minuit.print_all_minos`; use print() on the respective objects instead

Implementation
~~~~~~~~~~~~~~
- improved style of draw_contour, draw more contour lines
- increased default resolution for curves produced by `(draw_)mncontour`, `(draw_)contour`
- switched from internal copy of Minuit2 to including Minuit2 repository from GooFit
- build improvements for windows/msvc
- updated Minuit2 code to ROOT-v6.15/01 (compiler with C++11 support is now required to build iminuit)
- @henryiii added support for building Python-3.8 wheels

Documentation
~~~~~~~~~~~~~
- added iminuit logo
- added benchmark section
- expanded FAQ section
- updated basic tutorial to show how parameter values can be fixed and released
- added tutorial about combining iminuit with automatic differentiation
- clarified the difference between `profile` and `mnprofile`, `contour` and `mncontour`
- fixed broken URLs for external documents
- many small documentation improvements to increase consistency

1.3.8 (October 17, 2019)
------------------------

- fixed internal plotting when Minuit.from_array_func is used
- documentation updates
- reproduceable build

1.3.7 (June 12, 2019)
---------------------

- fixed wheels support
- fixed failing tests on some platforms
- documentation updates

1.3.6 (May 19, 2019)
--------------------

- fix for broken display of Jupyter notebooks on Github when iminuit output is shown
- replaced brittle and broken REPL diplay system with standard _repr_html_ and friends
- wheels support
- support for pypy-3.6
- documentation improvements
- new integration tests to detect breaking changes in the API

1.3.5 (May 16, 2019) [do not use]
---------------------------------

- release with accidental breaking change in the API, use 1.3.6

1.3.4 (May 16, 2019) [do not use]
---------------------------------

- incomplete release, use 1.3.6

1.3.3 (August 13, 2018)
-----------------------

- fix for broken table layout in print_param() and print_matrix()
- fix for missing error report when error is raised in user function
- fix of printout when ipython is used as a shell
- fix of slow convergence when analytical gradient is provided
- improved user guide with more detail information and improved structure

1.3.2 (August 5, 2018)
----------------------

- allow fixing parameter by setting limits (x, x) with some value x
- better defaults for maxcall arguments of hesse() and minos()
- nicer output for print_matrix()
- bug-fix: covariance matrix reported by iminuit was broken when some parameters were fixed
- bug-fix: segfault when something in PythonCaller raised an exception

1.3.1 (July 10, 2018)
---------------------

- fixed failing tests when only you installed iminuit with pip and don't
  have Cython installed

1.3 (July 5, 2018)
------------------

- iminuit 1.3 is a big release, there are many improvements. All users are encouraged to update.
- Python 2.7 as well as Python 3.5 or later are supported, on Linux, MacOS and Windows.
- Source packages are available for PyPI/pip and we maintain binary package for conda (see :ref:`install`).
- The bundled Minuit C++ library has been updated to the latest version (takend from ROOT 6.12.06).
- The documentation has been mostly re-written. To learn about `iminuit` and all the new features,
  read the :ref:`tutorials`.
- Numpy is now a core dependency, required to compile iminuit.
- For Numpy users, a second callback function interface and a ``Minuit.from_array_func`` constructor
  was added, where the parameters are passed as an array.
- Results are now also available as Numpy arrays, e.g. ``np_values``, ``np_errors`` and ``np_covariance``.
- A wrapper function ``iminuit.minimize`` for the MIGRAD optimiser was added,
  that has the same arguments and return value format as ``scipy.optimize.minimize``.
- Support for analytical gradients has been added, users can pass a ``grad`` callback function.
  This works, but for unknown reasons doesn't lead to performance improvements yet.
  If you can help debug or fix this issue, please comment `here <https://github.com/scikit-hep/iminuit/issues/252>`__.
- Several issues have been fixed.
  A complete list of issues and pull requests that went into the 1.3 release is
  `here <https://github.com/scikit-hep/iminuit/milestone/4?closed=1>`__.

Previously
----------

- For iminuit releases before v1.3, we did not fill a change log.
- To summarise: the first iminuit release was v1.0 in Dec 2012.
  In 2013 there were several releases, and in Jan 2014 the v1.1.1 release
  was made. After that development was mostly inactive, except for the
  v1.2 release in Nov 2015.
- The release history is available here: https://pypi.org/project/iminuit/#history
- The git history and pull requests are here: https://github.com/scikit-hep/iminuit
