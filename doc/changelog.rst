.. include:: bibliography.txt

.. _changelog:

Changelog
=========

2.28.0 (August 01, 2024)
------------------------
- Add name argument to all cost functions (`#1017 <https://github.com/scikit-hep/iminuit/pull/1017>`_)
- Fix leastsquares for functions with more than two arguments (`#1016 <https://github.com/scikit-hep/iminuit/pull/1016>`_)
- Drop support for python-3.8 (`#1015 <https://github.com/scikit-hep/iminuit/pull/1015>`_)
- Updating docs and links and ci for building docs (`#1014 <https://github.com/scikit-hep/iminuit/pull/1014>`_)
- Ci: try to make faster (`#996 <https://github.com/scikit-hep/iminuit/pull/996>`_)

2.27.0 (July 30, 2024)
----------------------
- Fix odr issue in type_caster (`#1013 <https://github.com/scikit-hep/iminuit/pull/1013>`_)
- More robust unified fitting in minuit.migrad and minuit.mnprofile (`#1009 <https://github.com/scikit-hep/iminuit/pull/1009>`_)
- Cure error when extremes cannot be found (`#1011 <https://github.com/scikit-hep/iminuit/pull/1011>`_)
- Fix for visibledeprecationwarning, useful annotated types, support tuple annotation (`#1004 <https://github.com/scikit-hep/iminuit/pull/1004>`_)
- Ci: using uv with cibuildwheel (`#999 <https://github.com/scikit-hep/iminuit/pull/999>`_)

2.26.0 (June 03, 2024)
----------------------
- Ci: add github artifact attestations to package distribution (`#993 <https://github.com/scikit-hep/iminuit/pull/993>`_)
- Ci: macos-latest is changing to macos-14 arm runners (`#989 <https://github.com/scikit-hep/iminuit/pull/989>`_)
- Added new tutorial on fitting correlated data (`#987 <https://github.com/scikit-hep/iminuit/pull/987>`_)
- Update to root master (`#986 <https://github.com/scikit-hep/iminuit/pull/986>`_)
- Check for odr violations and fix odr violation (`#985 <https://github.com/scikit-hep/iminuit/pull/985>`_)
- Improve comments on coverage (`#984 <https://github.com/scikit-hep/iminuit/pull/984>`_)
- Use modern ruff config & ruff fmt (`#978 <https://github.com/scikit-hep/iminuit/pull/978>`_)
- Bump the actions group with 3 updates (`#979 <https://github.com/scikit-hep/iminuit/pull/979>`_)
- Support numpy 2 (`#977 <https://github.com/scikit-hep/iminuit/pull/977>`_)

2.25.2 (February 09, 2024)
--------------------------
- Update to latest root (`#970 <https://github.com/scikit-hep/iminuit/pull/970>`_)
- Move documentation to Github pages (`#969 <https://github.com/scikit-hep/iminuit/pull/969>`_)

2.25.1 (February 08, 2024)
--------------------------
- Fix LeastSquares.visualize for models that accept parameter array (`#968 <https://github.com/scikit-hep/iminuit/pull/968>`_)
- Update benchmark to root-6.30 (`#967 <https://github.com/scikit-hep/iminuit/pull/967>`_)
- Improve docs for make_with_signature (`#963 <https://github.com/scikit-hep/iminuit/pull/963>`_)

2.25.0 (January 31, 2024)
-------------------------
- Approximate cdf from pdf (`#950 <https://github.com/scikit-hep/iminuit/pull/950>`_)
- Fix test that requires scipy and raise error on invalid value for use_pdf (`#962 <https://github.com/scikit-hep/iminuit/pull/962>`_)
- Fix docstring parsing (`#953 <https://github.com/scikit-hep/iminuit/pull/953>`_)
- Fix use of removed array rules in test (`#952 <https://github.com/scikit-hep/iminuit/pull/952>`_)
- Benchmark update to root 6.30 (`#951 <https://github.com/scikit-hep/iminuit/pull/951>`_)
- Fix: include debug info on failures (`#946 <https://github.com/scikit-hep/iminuit/pull/946>`_)
- Warn on errordef override (`#937 <https://github.com/scikit-hep/iminuit/pull/937>`_)
- Cost gradient support (`#936 <https://github.com/scikit-hep/iminuit/pull/936>`_)

2.24.0 (August 15, 2023)
------------------------
- Iteration limit in smart sampling to fix behavior for step functions (`#928 <https://github.com/scikit-hep/iminuit/pull/928>`_)
- Clarify meaning of 2d contours in minuit.draw_mnmatrix (`#927 <https://github.com/scikit-hep/iminuit/pull/927>`_)
- Support interval type and check compatibility with pydantic (`#922 <https://github.com/scikit-hep/iminuit/pull/922>`_)

2.23.0 (July 28, 2023)
----------------------
- Fix ``CostSum.visualize`` bug (`#918 <https://github.com/scikit-hep/iminuit/pull/918>`_)
- Fix ``_safe_log`` on systems which use 32 bit floats (`#915 <https://github.com/scikit-hep/iminuit/pull/915>`_), (`#919 <https://github.com/scikit-hep/iminuit/pull/919>`_)
- Skip test_interactive and friends when ipywidgets is not installed (`#917 <https://github.com/scikit-hep/iminuit/pull/917>`_)
- Turn negative zero into positive zero in ``pdg_format`` (`#916 <https://github.com/scikit-hep/iminuit/pull/916>`_)
- Remove warning in ``test_cost.py`` by using ``pytest.warns`` instead of ``pytest.raises`` (`#914 <https://github.com/scikit-hep/iminuit/pull/914>`_)

2.22.0 (June 22, 2023)
----------------------
- Hide confusing notes in docs: "not to be initialized by users." (`#906 <https://github.com/scikit-hep/iminuit/pull/906>`_)
- Fix: sdist size reduction (`#904 <https://github.com/scikit-hep/iminuit/pull/904>`_)
- Fix tests on freebsd, remove util._jacobi (`#903 <https://github.com/scikit-hep/iminuit/pull/903>`_)
- Update conclusions after the fix from jonas (`#899 <https://github.com/scikit-hep/iminuit/pull/899>`_)
- Use scikit-build-core (`#812 <https://github.com/scikit-hep/iminuit/pull/812>`_)
- Fix: make nograd not use grad at all in automatic diff doc (`#895 <https://github.com/scikit-hep/iminuit/pull/895>`_)
- Bump pypa/cibuildwheel from 2.12.3 to 2.13.0 (`#897 <https://github.com/scikit-hep/iminuit/pull/897>`_)
- Add minuit.fixto (`#894 <https://github.com/scikit-hep/iminuit/pull/894>`_)
- New benchmarks (`#893 <https://github.com/scikit-hep/iminuit/pull/893>`_)
- Make covariance fields in display easier to understand (`#891 <https://github.com/scikit-hep/iminuit/pull/891>`_)
- Improve docs (`#890 <https://github.com/scikit-hep/iminuit/pull/890>`_)
- Bump pypa/cibuildwheel from 2.12.1 to 2.12.3 (`#886 <https://github.com/scikit-hep/iminuit/pull/886>`_)
- Update fcn.hpp (`#889 <https://github.com/scikit-hep/iminuit/pull/889>`_)
- Fix-typo-in-basic (`#888 <https://github.com/scikit-hep/iminuit/pull/888>`_)
- Add leastsquares.pulls and leastsquares.prediction (`#880 <https://github.com/scikit-hep/iminuit/pull/880>`_)
- Better log-spacing detection (`#878 <https://github.com/scikit-hep/iminuit/pull/878>`_)
- Roofit tutorials (`#877 <https://github.com/scikit-hep/iminuit/pull/877>`_)
- Rename keyword nbins to bins in unbinnedcost.visualize (`#876 <https://github.com/scikit-hep/iminuit/pull/876>`_)
- Ignore missing matplotlib when calling minuit._repr_html_() (`#875 <https://github.com/scikit-hep/iminuit/pull/875>`_)
- Forward kwargs in minuit.visualize to plotting function (`#874 <https://github.com/scikit-hep/iminuit/pull/874>`_)
- Add hide_modules and deprecated_parameters (`#873 <https://github.com/scikit-hep/iminuit/pull/873>`_)
- Update progressbar (`#872 <https://github.com/scikit-hep/iminuit/pull/872>`_)
- Better ruff settings and adjustments, improvements to readme (`#871 <https://github.com/scikit-hep/iminuit/pull/871>`_)
- Use unicodeitplus instead of unicodeit to render latex as unicode (`#868 <https://github.com/scikit-hep/iminuit/pull/868>`_)
- Add roofit tutorial (`#867 <https://github.com/scikit-hep/iminuit/pull/867>`_)
- Improve error message for cost function (`#863 <https://github.com/scikit-hep/iminuit/pull/863>`_)
- Experimental mncontour algorithm (`#861 <https://github.com/scikit-hep/iminuit/pull/861>`_)
- Integer as variable (`#860 <https://github.com/scikit-hep/iminuit/pull/860>`_)
- Replace flake8 with ruff (`#859 <https://github.com/scikit-hep/iminuit/pull/859>`_)
- Add basic latex display support if unicodeit is installed (`#858 <https://github.com/scikit-hep/iminuit/pull/858>`_)

2.21.3 (April 03, 2023)
-----------------------
- Fix template input modification bug in template class (`#856 <https://github.com/scikit-hep/iminuit/pull/856>`_)
- Better docs for limits from annotated model parameters (`#853 <https://github.com/scikit-hep/iminuit/pull/853>`_)

2.21.2 (March 19, 2023)
-----------------------
- Fix CITATION.CFF

2.21.1 (March 18, 2023)
-----------------------
- Fix string annotations (`#849 <https://github.com/scikit-hep/iminuit/pull/849>`_)
- Specifiy minimum required numpy version (`#848 <https://github.com/scikit-hep/iminuit/pull/848>`_)

2.21.0 (March 03, 2023)
-----------------------
- Fix of matrix_format (`#843 <https://github.com/scikit-hep/iminuit/pull/843>`_)
- Support annotated model parameters (`#839 <https://github.com/scikit-hep/iminuit/pull/839>`_)
- Visualize fit in minuit._repr_html_ (`#838 <https://github.com/scikit-hep/iminuit/pull/838>`_)

2.20.0 (February 13, 2023)
--------------------------
- Fix coverage, typing, template for 2d models (`#836 <https://github.com/scikit-hep/iminuit/pull/836>`_)
- Template mix (`#835 <https://github.com/scikit-hep/iminuit/pull/835>`_)
- Docs: use 'req@url' syntax to install from remote vcs (`#834 <https://github.com/scikit-hep/iminuit/pull/834>`_)

2.19.0 (February 10, 2023)
--------------------------
- Remove setup.cfg (`#833 <https://github.com/scikit-hep/iminuit/pull/833>`_)
- Fix typo "probility" to "probability" in minos doc (`#832 <https://github.com/scikit-hep/iminuit/pull/832>`_)
- Prediction for all binned likelihoods (`#831 <https://github.com/scikit-hep/iminuit/pull/831>`_)
- Template prediction (`#820 <https://github.com/scikit-hep/iminuit/pull/820>`_)
- Upgrade old build and ci (`#817 <https://github.com/scikit-hep/iminuit/pull/817>`_)
- Add configurable number of bins for extended/unbinnednll (`#815 <https://github.com/scikit-hep/iminuit/pull/815>`_)

2.18.0 (December 14, 2022)
--------------------------
- Add more checks for gradients (`#810 <https://github.com/scikit-hep/iminuit/pull/810>`_)
- Bump pypa/cibuildwheel from 2.10.2 to 2.11.2 (`#808 <https://github.com/scikit-hep/iminuit/pull/808>`_)
- Added visualize function to minuit (`#799 <https://github.com/scikit-hep/iminuit/pull/799>`_)
- Move tutorials (`#806 <https://github.com/scikit-hep/iminuit/pull/806>`_)

2.17.0 (September 26, 2022)
---------------------------
- Add python 3.11, drop 3.6 (`#792 <https://github.com/scikit-hep/iminuit/pull/792>`_)
- Add Template fitting method from Argüelles, Schneider, Yuan (`#789 <https://github.com/scikit-hep/iminuit/pull/789>`_)
- Deprecate `iminuit.cost.BarlowBeestonLite` in favor of `iminuit.cost.Template`

2.16.0 (August 16, 2022)
------------------------
- Root update (`#786 <https://github.com/scikit-hep/iminuit/pull/786>`_)
- Fix corner case treatment of linear constraint (`#785 <https://github.com/scikit-hep/iminuit/pull/785>`_)
- Comparison with broadcasting (`#784 <https://github.com/scikit-hep/iminuit/pull/784>`_)
- Fix typing issues and enable mypy in pre-commit (`#783 <https://github.com/scikit-hep/iminuit/pull/783>`_)
- Make fixedview act as mask for other views (`#781 <https://github.com/scikit-hep/iminuit/pull/781>`_)

2.15.2 (August 03, 2022)
------------------------
- Improve docs for minimize (`#777 <https://github.com/scikit-hep/iminuit/pull/777>`_)
- Fix for minuit.interactive when using an array-based function (`#776 <https://github.com/scikit-hep/iminuit/pull/776>`_)

2.15.1 (July 28, 2022)
----------------------
- Fix pickling of all builtin cost functions (`#773 <https://github.com/scikit-hep/iminuit/pull/773>`_)

2.15.0 (July 27, 2022)
----------------------
- Smart sampling (`#769 <https://github.com/scikit-hep/iminuit/pull/769>`_)
- Enhance interactive (`#768 <https://github.com/scikit-hep/iminuit/pull/768>`_)

2.14.0 (July 25, 2022)
----------------------
- Interactive fitting (`#766 <https://github.com/scikit-hep/iminuit/pull/766>`_)

2.13.0 (July 17, 2022)
----------------------
- Interpolated mncontour (`#764 <https://github.com/scikit-hep/iminuit/pull/764>`_)
- Added mnmatrix plot (`#763 <https://github.com/scikit-hep/iminuit/pull/763>`_)
- Close mncontour for convenience (`#761 <https://github.com/scikit-hep/iminuit/pull/761>`_)
- Update tutorials (`#760 <https://github.com/scikit-hep/iminuit/pull/760>`_)

2.12.2 (July 15, 2022)
----------------------
- fix a bug in error heuristic when parameters have negative values and prevent assigning negative values to errors (`#759 <https://github.com/scikit-hep/iminuit/pull/759>`_)

2.12.1 (July 1, 2022)
---------------------

Fixes
~~~~~
- ``cost.BarlowBeestonLite``: method "hpd" has been modified to fix performance in cases where bins are not dominated by a single template

2.12.0 (June 21, 2022)
----------------------

New features
~~~~~~~~~~~~
- New cost function ``cost.BarlowBeestonLite`` for template fits with correct uncertainty propagation for templates obtained from simulation or sWeighted data, written together with @AhmedAbdelmotteleb
- Formerly private chi2 utility cost functions (``cost.poisson_chi2``, etc.), are now part of public API
- Support custom grid in ``Minuit.profile``, ``iminuit.mncontour``, ``iminuit.contour``
- Handle common CL values in ``Minuit.mnprofile`` and ``Minuit.mncontour`` without scipy

Fixes
~~~~~
- Skip tests that use ``np.float128`` on platforms where this type is not supported
- ``Minuit.valid`` now returns ``False`` if EDM is NaN
- ``subtract_min`` setting is no longer ignored by ``Minuit.draw_contour``

Documentation
~~~~~~~~~~~~~
- New study about template fits

Other
~~~~~
- ``Minuit`` no longer warns when a function is used that has no ``errordef``
  attribute and ``Minuit.errordef`` is not explicitly set. The function is
  assumed to be chi-square-like up to an arbitrary constant, unless ``errordef``
  is explicitly set to something else.
- More type correctness in API, better hiding of private objects in library
- Option to use external pybind11, by @henryiii
- Update to pybind11-v2.9.2, by @henryiii
- Update to root-v6-25-02 plus our patches
- Several minor cleanups and internal tool updates, by @henryiii

2.11.2 (March 28, 2022)
-----------------------

Other
~~~~~
- Fixed wording in cost function tutorial

2.11.1 (March 27, 2022)
-----------------------

Fixes
~~~~~
- Fixed a failure of ``util.make_with_signature`` in some situations

Other
~~~~~
- Raise numpy.VisibleDeprecationWarning instead of warnings.DeprecationWarning
- ``util.propagate`` is deprecated in favour of ``jacobi.propagate`` from the jacobi library

2.11.0 (March 27, 2022)
-----------------------

New features
~~~~~~~~~~~~
- All builtin cost functions now support multidimensional data
- ``Matrix.to_dict`` was added for symmetry with ``BasicValueView.to_dict``
- For long-running fits, total runtime is now shown in ``FMin`` display and total runtime can be accessed via property ``FMin.time``

Fixes
~~~~~
- In binned fits when ``ndof`` is zero, show ``reduced chi2 = nan`` in the ``FMin`` display instead of raising a ZeroDivisionError

Documentation
~~~~~~~~~~~~~
- Tutorials and studies are now listed separately
- Tutorial for fits of multivariate data were added
- The cost function tutorial was improved
- Studies in regard to performance were added, including a comparison with RooFit

2.10.0 (March 4, 2022)
----------------------

New features
~~~~~~~~~~~~
- ``UnbinnedNLL`` and ``ExtendedUnbinnedNLL`` now support models that predict the
  ``logpdf`` instead of the ``pdf`` when the extra keyword ``log=True`` is set; when
  it is possible, using the ``logpdf`` instead of the ``pdf`` for fitting is faster
  and numerically more stable

2.9.0 (January 7, 2022)
-----------------------

Fixes
~~~~~
- ``Minuit.draw_mncontour`` now works with matplotlib >= 3.5
- Builtin cost functions now work correctly when the mask is set and data is updated on
  the existing cost function

Performance
~~~~~~~~~~~
- Builtin cost functions are now more performant when used with weighted binned data

Other
~~~~~
- Wheels for Python 3.10, by @henryiii
- Bump pybind11 to 2.9.0, by @henryiii

2.8.4 (October 11, 2021)
------------------------

Fixes
~~~~~
- Pickling of ``util.Matrix`` resulted in incomplete state after unpickling, which would
  cause an exception when you tried to print the matrix

Documentation
~~~~~~~~~~~~~
- New tutorial on fitting PDFs that depend on a conditional variable
- Fixed JAX tutorial, adapting to change in their interface
- Extended documentation of cost functions

2.8.3 (September 3, 2021)
-------------------------

New features
~~~~~~~~~~~~
- ``util.propagate`` now discriminates between diverging derivates (using the value NaN
  for the derivate) and non-converging derivatives (using the best value computed so far
  for the derivative)

Documentation
~~~~~~~~~~~~~
- Fixes for faulty LaTeX rendering in some tutorials

Other
~~~~~
- Support cross-compiling of ARM on Conda, by @henryiii

2.8.2 (August 15, 2021)
-----------------------

Fixes
~~~~~
- ``Minuit.draw_mncontour`` can now be used by passing a single float to keyword ``cl``,
  in addition to passing a list of floats
- Use ``pybind11::ssize_t`` everywhere instead of non-standard ``ssize_t`` to fix
  compilation against Python-3.10 on Windows, co-authored with @cgohlke

Documentation
~~~~~~~~~~~~~
- Docstring improved for ``Minuit.mncontour``, advice added on how to draw closed curve
  with points returned by ``Minuit.mncontour``
- Docstring improved for ``Minuit.draw_mncontour``, parameters and returned objects are
  now properly documented

2.8.1 (August 4, 2021)
----------------------

Other
~~~~~
- @henryiii added Apple Silicon wheels
- @odidev added Linux aarch64 wheels

2.8.0 (July 25, 2021)
---------------------

Minor API change
~~~~~~~~~~~~~~~~
- ``Minuit.mncontour`` now raises ``RuntimeError`` instead of ``ValueError`` if it is
  not called at a valid minimum point

New features
~~~~~~~~~~~~
- ``Minuit.mncontour`` can now be called at any point without running a minimiser before,
  similar to ``Minuit.minos``

Fixes
~~~~~
- ``Minuit.mncontour`` used to fail if called twice in a row

2.7.0 (July 4, 2021)
--------------------

Minor API change
~~~~~~~~~~~~~~~~
- If ``Minuit.hesse`` is called when ``Minuit.fmin`` is ``None``, an instance
  ``Minuit.fmin`` is now created. If Hesse fails, the code does not raise an exception
  anymore, since now the error state can be accessed as usual from the ``Minuit.fmin``
  object. Users who relied on the exception should check ``Minuit.fmin`` from now on.

New features
~~~~~~~~~~~~
- ``Minuit.scipy`` can be used to minimise with SciPy algorithms. These may succeed
  when ``Minuit.migrad`` fails and support additional features. Some algorithms
  allow one to pass a function that returns the Hessian matrix (which may be computed
  analytically or via automatic differentiation provided by other libraries like JAX).
  Other algorithms support minimisation under arbitrary non-linear constraints.
- ``util.FMin`` has new html/text representations; the field ``Valid parameters`` was
  removed, a title with the name of the minimisation method was added
- ``Minuit.tol`` now accepts the values 0 and ``None``, the latter resets the default
- Builtin cost functions now support models that return arrays in long double precision
  ``float128``. In this case, all computations inside the cost function are also done in
  higher precision.
- Builtin cost functions now raise a warning if the user-defined model does not return
  a numpy array

Fixes
~~~~~
- Calling ``Minuit.migrad`` with a call limit under some circumstances used much more
  calls than expected and did not report that the call limit was reached
  (patch submitted to ROOT)
- ``Minuit.hesse`` no longer sets the status of the FunctionMinimum unconditionally
  to valid if it was invalid before
- Repeated calls to ``Minuit.hesse`` no longer accumulate calls and eventually exhaust
  the call limit, the call counter is now properly reset
- Calling ``Minuit.minos`` repeatedly now does not recompute the Hessian and avoids
  a bug that used to exhaust the call limit before in this case

Documentation
~~~~~~~~~~~~~
- Tutorial notebooks are now fully integrated into the HTML documentation
- A tutorial on using constrained minimisation from SciPy for HEP task was added

Other
~~~~~
- ``util.BasicView`` is now a proper abstract base class

2.6.1 (May 13, 2021)
--------------------

Fixes
~~~~~
- Calling ``Minuit.fixed[...] = False`` on parameter that was not fixed before
  lead to undefined behaviour in Minuit2 C++ code (patch submitted to ROOT)

Other
~~~~~
- Upgrade Minuit2 C++ code to latest ROOT master with simplified internal class
  structure and class tags replaced with enums

2.6.0 (May 2, 2021)
-------------------

New features
~~~~~~~~~~~~
- Builtin cost functions now report the number of data points with the attribute
  ``Cost.ndata``
- New attribute ``Minuit.ndof`` returns the degrees of freedom if the cost function
  reports it or NaN
- New attribute ``FMin.reduced_chi2`` to report the reduced chi2 of the fit; returns
  NaN if the reduced chi2 cannot be computed for the cost function, in case of unbinned
  maximum-likelihood or when the attribute ``Cost.ndata`` is missing

2.5.0 (April 30, 2021)
----------------------

New features
~~~~~~~~~~~~
- ``util.merge_signatures`` added based on ``merge_user_func`` from ``probfit``,
  by @mbaak
- ``util.make_with_signature`` added to create new functions with renamed arguments
- ``util.BasicView.to_dict`` added, by @watsonjj
- ``util.BasicView`` and ``util.Matrix`` now supports element selection with sequences
  like ``numpy.ndarray``
- ``util.propagate`` to error propagate covariance matrices from one vector space to
  another (Jacobi matrix is computed numerically)

Fixes
~~~~~
- ``util.BasicView`` now supports slices of the form ``a[:len(a)]`` or ``a[:M]`` with
  ``M > len(a)`` like other Python containers
- ``util.Matrix`` now returns a square matrix when it is used with a slice or item
  selection
- Missing comma in BibTeX entry shown in CITATION.rst, by Ludwig Neste

Other
~~~~~
- ``util.describe`` returns list instead of tuple

Documentation
~~~~~~~~~~~~~
- Better docstring for ``util.FMin``
- New tutorial on how to do simultaneous fits / adding likelihoods, by @watsonjj
- New tutorial on how to use builtin cost function
- New tutorial about how to draw error bands around fitted curves

2.4.0 (February 10, 2021)
-------------------------

New features
~~~~~~~~~~~~
- ``minimize``
   - Keyword ``method`` now accepts "migrad" and "simplex"
   - Keyword ``option`` now supports keyword "stra" to set ``Minuit.strategy``
   - ``OptimizeResult.message`` now states if errors are not reliable
- ``Minuit`` now supports functions wrapped with ``functools.partial``, by @jnsdrtlf

Other
~~~~~
- Upgrade Minuit2 C++ code in ROOT to latest version with following improvements

   - improvement of seed when using an analytical gradient
   - fix of last minimum state added twice to vector of minimum states in some cases
     (no impact for iminuit users, but saves a bit of memory)

- Documentation improvements
- Updated tutorial about automatic differentiation, added comparison of ``numba.njit``
  and ``jax.jit``

2.3.0 (January 24, 2021)
------------------------

New features
~~~~~~~~~~~~
- ``cost.BinnedNLL`` and  ``cost.ExtendedBinnedNLL`` now support
  weighted binned data

Bug-fixes
~~~~~~~~~
- ``FMin.edm_goal`` now remains unchanged if ``Minuit.hesse`` is run after
  ``Minuit.migrad``

Other
~~~~~
- Update to cibuildwheels-1.8.0 and workflow simplification, by @henryiii

2.2.1 (December 22, 2020)
-------------------------

Minor improvements
~~~~~~~~~~~~~~~~~~
- ``Minuit.profile``, ``Minuit.mnprofile``, ``Minuit.contour``, ``Minuit.draw_profile``,
  ``Minuit.draw_mnprofile``, and ``Minuit.draw_contour`` can now be called with
  ``subtract_min=True`` even if ``Minuit.fmin`` is None
- ``__version__`` now also displays the ROOT version of the C++ Minuit2 library
- Support for adding constant numbers to cost functions, this allows you to write
  ``sum(cost1, cost2, ...)`` and may be useful to subtract a constant bias from the
  cost

Other
~~~~~
- Documentation improvements

   - Further transition to numpydoc
   - Clarified that iminuit is based on ROOT code
   - List full iminuit version including ROOT version in docs

- Added type hints to many interfaces (incomplete)
- Renamed ``_minuit`` to ``minuit``, making the module public
- Renamed ``_minimize`` to ``minimize``, making the module public
- pydocstyle added to pre-commit checks

2.2.0 (December 20, 2020)
-------------------------

New features
~~~~~~~~~~~~
- Cost functions in ``cost`` are now additive, creating a new cost function with the
  union of parameters that returns the sum of the results of the individual cost functions
- ``cost.NormalConstraint`` was added as a means to add soft constraints on a
  parameter, can also be used to set up a covariance matrix between several parameters

Other
~~~~~
- Documentation improvements, started transition to numpydoc

2.1.0 (December 18, 2020)
-------------------------

New features
~~~~~~~~~~~~
- Minuit object is now pickle-able and copy-able
- More efficient internal conversion between Python objects and ``std::vector<double>``
- ``Minuit.minos`` can now be called without calling ``Minuit.migrad`` first, which allows
  one to use an external minimiser to find a minimum and then compute Minos errors for it

Bug-fixes
~~~~~~~~~
- User-supplied gradient functions that return a ``torch.Tensor`` now work again
- Matrix display now shows numbers correctly even if entries differ in magnitude

Other
~~~~~
- Unit tests are included again in sdist package
- ``Minuit.grad`` now returns ``numpy.ndarray`` instead of a ``list``
- Fixes for ``conda`` builds on Windows platform with ``msvc``, by @henryiii
- Updated and unified documentation on how to cite iminuit
- ``print()`` applied to ``Minuit.params``, ``Minuit.merrors``, ``Minuit.covariance``,
  ``Minuit.fmin`` now returns the pretty text version again instead of the
  ``repr`` version
- Update to pybind11 v2.6.1

2.0.0 (December 7, 2020)
------------------------

This is a breaking change for  Interface that was deprecated in 1.x has been removed. In addition, breaking changes were made to the interface to arrive at a clean minimal state that is easier to learn, safer to use, and ready for the long-term future. **To keep existing scripts running, pin your major iminuit version to <2**, i.e. ``pip install 'iminuit<2'`` installs the 1.x series.

Under the hood, Cython was replaced with pybind11 to generate the bindings to the C++ Minuit2 library. This simplified the code considerably (Cython is bad at generating correct C++ bindings, while it is a breeze with pybind11).

Removed and changed interface (breaking changes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``Minuit.__init__``

  - Keywords ``error_*``, ``fix_*``, and ``limit_*`` were removed; assign to ``Minuit.errors``, ``Minuit.fixed``, and ``Minuit.limits`` to set initial step sizes, fix parameters, and set limits
  - Keyword ``pedantic`` was removed; parameters must be initialised with values now or an error is raised
  - Keyword ``errordef`` was removed; assign to ``Minuit.errordef`` to set the error definition of the cost function or better create an attribute called ``errordef`` on the cost function, Minuit uses this attribute if it exists
  - Keyword ``throw_nan`` was removed; assign to ``Minuit.throw_nan`` instead
  - Keyword ``print_level`` was removed; assign to ``Minuit.print_level`` instead
  - Setting starting values with positional parameters is now allowed, e.g. ``Minuit(my_fcn, 1, 2)`` initialises the first parameters to 1 and the second to 2
  - Keyword ``use_array_call`` was removed; call type is inferred from the initialisation value, if it is a sequence, the array call is used (see next item below)

- ``Minuit.from_array_func`` was removed; use ``Minuit(some_numpy_function, starting_array)`` instead
- ``Minuit.args`` was removed, use ``Minuit.values[:]`` to get the current parameter values as a tuple
- ``Minuit.values``

  - Now behaves like an array instead of like a dict, i.e. methods like ``keys()`` and ``items()`` are gone and ``for x in minuit.values`` iterates over the values
  - Item access via index and via parameter name is supported, e.g. ``minuit.values[0]`` and ``minuit.values["a"]`` access the value for the first parameter "a"
  - Broadcasting is supported, e.g. ``minuit.values = 0`` sets all parameter values to 0
  - Slicing is supported for setting and getting several parameter values at once

- ``Minuit.errors``: see changes to ``Minuit.values``
- ``Minuit.fixed``: see changes to ``Minuit.values``
- ``Minuit.migrad``

  - Keyword ``resume`` was removed; use ``Minuit.reset`` instead
  - Keyword ``precision`` was removed; use ``Minuit.precision`` instead
  - Return value is now ``self`` instead of ``self.fmin, self.params``

- ``Minuit.hesse``: Return value is now ``self`` instead of ``self.params``
- ``Minuit.minos``

  - Now accepts more than one positional argument (which must be parameter names) and computes Minos errors for them
  - Return value is now ``self`` instead of ``self.merrors``
  - ``sigma`` keyword replaced with ``cl`` to set confidence level (requires scipy)

- ``Minuit.mncontour`` and ``Minuit.draw_mncontour``

  - ``sigma`` keyword replaced with ``cl`` to set confidence level (requires scipy)
  - ``numpoints`` keyword replaced with ``size``
  - Keyword arguments are keyword-only
  - Return value is reduced to just the contour points as a numpy array

- ``Minuit.mnprofile`` and ``Minuit.draw_mnprofile``

  - ``sigma`` keyword replaced with ``cl`` to set confidence level (requires scipy)
  - ``numpoints`` keyword replaced with ``size``
  - Keyword arguments are keyword-only

- ``Minuit.profile`` and ``Minuit.draw_profile``

  - ``bins`` keyword replaced with ``size``
  - Keyword arguments are keyword-only

- ``Minuit.fitarg`` was removed; to copy state use ``m2.values = m1.values; m2.limits = m1.limits`` etc. (Minuit object may become copyable and pickleable in the future)
- ``Minuit.matrix`` was removed; see ``Minuit.covariance``
- ``Minuit.covariance`` instead of a dict-like class is now an enhanced subclass of numpy.ndarray (util.Matrix) with the features:

  - Behaves like a numpy.ndarray in numerical computations
  - Rich display of the matrix in ipython and Jupyter notebook
  - Element access via parameter names in addition to indices, e.g. Minuit.covariance["a", "b"] access the covariance of parameters "a" and "b"
  - ``Minuit.covariance.correlation()`` computes the correlation matrix from the covariance matrix and returns it
  - Has always full rank, number of rows and columns is equal to the number of parameters even when some are fixed; elements corresponding to fixed parameters are set to zero in the matrix

- ``Minuit.gcc`` was removed for lack of a known use-case (submit an issue if you need this, then it will come back)
- ``Minuit.is_clean_state`` was removed; use ``Minuit.fmin is None`` instead
- ``Minuit.latex_param`` was removed; LaTeX and other table formats can be produced by passing the output of ``minuit.params.to_table()`` to the external ``tabulate`` module available on PyPI
- ``Minuit.latex_initial_param`` was removed; see ``Minuit.latex_param``
- ``Minuit.latex_matrix`` was removed; LaTeX and other table formats can be produced by passing the output of ``minuit.covariance.to_table()`` to the external ``tabulate`` module available on PyPI
- ``Minuit.ncalls_total`` was replaced with ``Minuit.nfcn``
- ``Minuit.ngrads_total`` was replaced with ``Minuit.ngrad``
- ``Minuit.np_covariance`` is now obsolete and was removed; see ``Minuit.covariance``
- ``Minuit.np_matrix`` is now obsolete and was removed; see ``Minuit.covariance``
- ``Minuit.np_values`` was removed; use ``Minuit.values`` instead or ``np.array(m.values)``
- ``Minuit.np_errors`` was removed; use ``Minuit.errors`` instead or ``np.array(m.errors)``
- ``Minuit.np_merrors`` was removed; use ``Minuit.merrors`` or ``Minuit.params`` instead
- ``Minuit.use_array_call`` was removed, ``Minuit.fcn`` and ``Minuit.grad`` always require parameter values in form of sequences, e.g. ``minuit.fcn((1, 2))``
- ``util.FMin`` is now a data class with read-only attributes, the dict-like interface was removed (methods like ``keys()``, ``items()`` are gone)

  - ``tolerance`` attribute was replaced with ``edm_goal``, since the effect of ``tolerance`` varies for ``Minuit.migrad`` and ``Minuit.simplex``, ``edm_goal`` is the actual value of interest
  - Property ``nfcn`` is the total number of function calls so far
  - Property ``ngrad`` is the total number of gradient calls so far
  - ``ngrad_total`` was removed and replaced by ``ngrad``
  - ``nfcn_total`` was removed and replaced by ``nfcn``
  - ``up`` was removed and replaced by ``errordef`` (to have one consistent name)
  - ``util.MError`` is now a data class, dict-like interface was removed (see ``util.FMin``)
  - ``util.Param`` is now a data class, dict-like interface was removed (see ``util.FMin``)
- ``util.Matrix`` is now a subclass of a numpy.ndarray instead of a tuple of tuples
- ``util.InitialParamWarning`` was removed since it is no longer used
- ``util.MigradResult`` was removed since it is no longer used
- ``util.arguments_from_inspect`` was removed from the public interface, it lives on as a private function

New features
~~~~~~~~~~~~
- ``Minuit`` class

  - Now a class with `__slots__`; assigning to a non-existent attribute (e.g. because of a typo) now raises an error
  - Parameter names in Unicode are now fully supported, e.g. ``def fcn(α, β): ...`` works
  - New method ``simplex`` to minimise the function with the Nelder-Mead method
  - New method ``scan`` to minimise the function with a brute-force grid search (not recommended and infeasible for fits with more than a few free parameters)
  - New method ``reset`` reverts to the initial parameter state
  - New property ``limits``, an array-like view of the current parameter limits; allows to query and set limits with a behaviour analog to ``values``, ``errors`` etc.; broadcasting is supported, e.g. ``minuit.limits = (0, 1)`` makes all parameters bounded between 0 and 1 and ``minuit.limits = None`` removes all limits
  - New property ``precision`` to change the precision that Minuit assumes for internal calculations of derivatives
  - Support for calling numba-compiled functions that release the GIL (slightly more efficient already today and may be used in the future to compute derivatives in parallel)
  - Now pretty-prints itself in Jupyter notebooks and the ipython shell, showing the equivalent of ``Minuit.fmin``, ``Minuit.params``, ``Minuit.merrors``, ``Minuit.covariance``, whatever is available

- ``util.Param`` class

  - New attribute ``merror``, which either returns a tuple of the lower and upper Minos error or None
  - All attributes are now documented inline with docstrings which can be investigated with ``pydoc`` and ``help()`` in the REPL

- ``util.Params`` class

  - New method ``to_table``, which returns a format that can be consumed by the external ``tabulate`` Python module

- ``util.FMin`` class

  - New attribute ``ngrad`` which is the number of gradient calls so far
  - New attribute ``has_parameters_at_limit`` which returns True if any parameter values is close to a limit
  - All attributes are now documented inline with docstrings which can be investigated with ``pydoc`` and ``help()`` in the REPL

- ``util.Matrix`` class

  - New method ``to_table``, which returns a format that can be consumed by the external ``tabulate`` Python module
  - New method ``correlation``, which computes and returns the correlation matrix (also a ``util.Matrix``)

Bug-fixes
~~~~~~~~~
- Calling ``Minuit.hesse`` when all parameters were fixed now raises an error instead of producing a segfault
- Many life-time/memory leak issues in the iminuit interface code should be resolved now, even when there is an exception during the minimisation (there can still be errors in the underlying C++ Minuit2 library, which would have to be fixed upstream)

Other changes
~~~~~~~~~~~~~
- Several attributes were replaced with properties to avoid accidental overrides and to protect against assigning invalid input, e.g. ``Minuit.tol`` and ``Minuit.errordef`` only accept positive numbers
- Documentation update and clean up
- Logging messages from C++ Minuit2, which are produced when ``Minuit.print_level`` is set to 1 to 3 are now properly shown inside the notebook or the Python session instead of being printed to the terminal
- Assigning to ``Minuit.print_level`` changes the logging threshold for all current and future ``Minuit`` instances in the current Python session, this is not really desired but cannot be avoided since the C++ logger is a global variable
- docstring parsing for ``util.describe`` was rewritten; behaviour of ``describe`` for corner cases of functions with positional and variable number of positional and keyword arguments are now well-defined
- iminuit now has 100 % line coverage by unit tests

1.5.4 (November 21, 2020)
--------------------------

- Fixed broken sdist package in 1.5.3

1.5.3 (November 19, 2020)
--------------------------

Fixes
~~~~~
- Fixed a crash when throw_nan=True is used and the throw is triggered
- Add python_requires (`#496 <https://github.com/scikit-hep/iminuit/pull/496>`_) by @henryiii
- Fixed buggy display of text matrix if npar != 2 (`#493 <https://github.com/scikit-hep/iminuit/pull/493>`_)

Other
~~~~~
- Switch extern Minuit2 repo to official root repo (`#500 <https://github.com/scikit-hep/iminuit/pull/500>`_), ROOT state: a5d880a434
- Add ngrad and ngrad_total to FMin display, rename ncalls to nfcn_total (`#489 <https://github.com/scikit-hep/iminuit/pull/489>`_)
- Use __getattr__ to hide deprecated interface from Python help() (`#491 <https://github.com/scikit-hep/iminuit/pull/491>`_)
- Improvements to tutorials by @giammi56
- Show number of gradient calls in FMin display (if nonzero) instead of errordef value

Deprecated
~~~~~~~~~~
- Minuit.ncalls, use Minuit.nfcn instead
- Minuit.ngrads, use Minuit.ngrad instead

1.5.2 (September 24, 2020)
--------------------------

- Fixed regression of the convergence rate of ``Minuit.migrad`` for low precision cost
  functions by restoring a heuristic that calls Migrad several times if convergence is
  not reached on first try; made this heuristic configurable with `iterate` keyword
- Clarify in ``FMin`` display how the EDM convergence criterion uses the EDM goal

1.5.1 (September 20, 2020)
--------------------------

- Fixed mistake in *parameter at limit* warning, which did not report correctly if parameter was at the upper limit

1.5.0 (September 17, 2020)
---------------------------

New features
~~~~~~~~~~~~
- New more compact function minimum display with warning about parameters at limit
- Colours adjusted in HTML display to enhance contrast for people with color blindness
- Allow subclasses to use ``Minuit.from_array_func`` (`#467 <https://github.com/scikit-hep/iminuit/pull/467>`_) [contributed by @kratsg]
- Nicer tables on terminal thanks to unicode characters
- Wrapped functions' parameters are now correctly recognized [contributed by Gonzalo]
- Dark theme friendlier HTML style (`#481 <https://github.com/scikit-hep/iminuit/pull/481>`_) [based on patch by @l-althueser]

Bug-Fixes
~~~~~~~~~
- Fixed reported EDM goal for really small tolerances
- ``Minuit.np_merrors`` now works correctly when some parameters are fixed
- Fixed HTML display of Minuit.matrix when some diagonal elements are zero

Deprecated
~~~~~~~~~~
- Removed ``nsplit`` option from ``Minuit.migrad`` (`#462 <https://github.com/scikit-hep/iminuit/pull/462>`_)

1.4.9 (July, 18, 2020)
----------------------

- Fixes an error introduced in 1.4.8 in ``Minuit.minos`` when ``var`` keyword is used and at least one parameter is fixed

1.4.8 (July, 17, 2020)
----------------------

- Allow ``ncall=None`` in ``Minuit.migrad``, ``Minuit.hesse``, ``Minuit.minos``
- Deprecated ``maxcall`` argument in ``Minuit.minos``: use ``ncall`` instead

1.4.7 (July, 15, 2020)
----------------------

- Fixed: ``cost.LeastSquares`` failed when ``yerror`` is passed as list and mask is set

1.4.6 (July, 11, 2020)
----------------------

- Update to Minuit2 C++ code to ROOT v6.23-01
- Fixed: iminuit now reports an invalid fit if a cost function has only a maximum, not a minimum (fixed upstream)
- Loss function in ``cost.LeastSquares`` is now mutable
- Cost functions in ``cost`` now support value masks
- Documentation improvements
- Fixed a deprecation warning in ``Minuit.mnprofile``
- Binder now uses wheels instead of compiling current iminuit

1.4.5 (June, 25, 2020)
----------------------

- Improved pretty printing for Minos Errors object ``MErrors``
- Added docs for cost functions

1.4.4 (June, 24, 2020)
----------------------

- Reverted: create MnHesse C++ instance on the stack instead on the heap
- Added least-squares cost function and tests

1.4.3 (June, 24, 2020)
----------------------

Bug-fixes
~~~~~~~~~
- Fixed a bug where running ``Minuit.hesse`` after ``Minuit.migrad``, which would ignore any changes to parameters (fixing/releasing them, changing their values, ...)
- Fix number formatting issues in new quantities display
- Removed engineering suffixes again in favour of standard exponential notation

Deprecated
~~~~~~~~~~
- keyword `forced_parameters` in `Minuit.__init__` is deprecated, use `name`

Features
~~~~~~~~
- Added general purpose cost functions for binned and unbinned maximum-likelihood estimation (normal and so called extended)

Documentation
~~~~~~~~~~~~~
- Updated error computation tutorial
- New tutorial which demonstrates usage of cost functions

1.4.2 (June, 14, 2020)
----------------------

Hot-fix release to correct an error in ``Minuit.merrors`` indexing.

Documentation
~~~~~~~~~~~~~
- New tutorial about using Numba to parallelize and jit-compile cost functions

1.4.1 (June, 13, 2020)
----------------------

Mostly a bug-fix release, but also deprecates more old interface.

Bug-fixes
~~~~~~~~~
- Fixed a bug when displaying nans in rich displays

Deprecated
~~~~~~~~~~
- ``Minuit.minoserror_struct``: use ``Minuit.merrors``, which is now an alias for the former
- ``Minuit.merrors`` now accepts indices and parameter names, like `Minuit.values`, etc.

Features
~~~~~~~~
- Show engineering suffixes (1.23k, 123.4M, 0.8G, ...) in rich diplays instead of "scientific format", e.g. 1.23e-3
- New initial step heuristic, replaces pedantic warning about missing step sizes
- New initial value heuristic for parameters with limits

Documentation
~~~~~~~~~~~~~
- New tutorial about using Numba to parallelize and jit-compile cost functions

1.4.0 (June, 12, 2020)
----------------------

This release drops Python 2 support and modernizes the interface of iminuit's Minuit object to make it more pythonic. Outdated methods were deprecated and replaced with properties. Keywords in methods were made more consistent. The deprecated interface has been removed from the documentation, but is still there. Old code should still work (if not please file a bug report!).

Bug-fixes
~~~~~~~~~
- Fixed an exception in the rich display when results were NaN
- ``Minuit.migrad_ok()`` (now replaced by `Minuit.accurate`) now returns false if HESSE failed after MIGRAD and made the minimum invalid
- Running ``Minuit.hesse()`` now properly updates the function minimum
- Fixed incorrect ``hess_inv`` returned by ``minimize``
- Fixed duplicated printing of pedantic warning messages

Deprecated
~~~~~~~~~~
- ``Minuit.list_of_fixed_params()``, ``Minuit.list_of_vary_params()``: use ``Minuit.fixed``
- ``Minuit.migrad_ok()``: use ``Minuit.valid``
- ``Minuit.matrix_accurate()``: use ``Minuit.accurate``
- ``Minuit.get_fmin()``: use ``Minuit.fmin``
- ``Minuit.get_param_states()``: use ``Minuit.param``
- ``Minuit.get_initial_param_states()``: use ``Minuit.init_param``
- ``Minuit.get_num_call_fcn()``: use ``Minuit.ncalls_total``
- ``Minuit.get_num_call_grad()``: use ``Minuit.ngrads_total``
- ``Minuit.print_param_states()``: use ``print`` on ``Minuit.params``
- ``Minuit.print_initial_param_states()``: use ``print`` on ``Minuit.init_params``
- ``Minuit.hesse(maxcall=...)`` keyword: use ``ncall=...`` like in ``Minuit.migrad``
- ``Minuit.edm``: use ``Minuit.fmin.edm``

New features
~~~~~~~~~~~~
- iminuit now uses the PDG formatting rule for quantities with errors
- slicing and basic broadcasting support for ``Minuit.values``, ``Minuit.errors``, ``Minuit.fixed``, e.g. the following works: ``m.fixed[:] = True``, ``m.values[:2] = [1, 2]``
- ``Minuit.migrad(ncall=0)`` (the default) now uses MINUITs internal heuristic instead of a flat limit of 10000 calls
- ``minimize`` now supports the ``tol`` parameter
- `Minuit` now supports ``print_level=3``, which shows debug level information when MIGRAD runs
- Binder support and Binder badge for tutorial notebooks added by @matthewfeickert

Documentation
~~~~~~~~~~~~~
- New tutorials on error computation and on using automatic differentiation

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
- ``Minuit.draw_contour`` now accepts an integer for ``bound`` keyword as advertised in the docs
- fixed wrong EDM goal in iminuit reports, was off by factor 5 in some

Interface
~~~~~~~~~
- removed the undocumented keyword "args" in ``(draw_)contour``, ``(draw_)profile``
- removed misleading "show_sigma" keyword in ``draw_contour``
- deprecated ``Minuit.is_fixed``, replaced by ``Minuit.fixed`` property
- deprecated ``Minuit.set_strategy``, assign to ``Minuit.strategy`` instead
- deprecated ``Minuit.set_errordef``, assign to ``Minuit.errordef`` instead
- deprecated ``Minuit.set_print_level``, assign to ``Minuit.print_level`` instead
- deprecated ``Minuit.print_fmin``, ``Minuit.print_matrix``, ``Minuit.print_param``, ``Minuit.print_initial_param``, ``Minuit.print_all_minos``; use ``print`` on the respective objects instead
- added ``Minuit.nfit`` to get number of fitted parameters

Implementation
~~~~~~~~~~~~~~
- improved style of draw_contour, draw more contour lines
- increased default resolution for curves produced by ``(draw_)mncontour``, ``(draw_)contour``
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
- clarified the difference between ``profile`` and ``mnprofile``, ``contour`` and ``mncontour``
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
- replaced brittle and broken REPL diplay system with standard ``_repr_html_`` and friends
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
- The documentation has been mostly re-written. To learn about iminuit and all the new features,
  read the :ref:`tutorials`.
- Numpy is now a core dependency, required to compile iminuit.
- For Numpy users, a second callback function interface and a ``Minuit.from_array_func`` constructor
  was added, where the parameters are passed as an array.
- Results are now also available as Numpy arrays, e.g. ``np_values``, ``np_errors`` and ``np_covariance``.
- A wrapper function ``minimize`` for the MIGRAD optimiser was added,
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
