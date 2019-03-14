iminuit
=======

.. image:: https://travis-ci.com/HDembinski/iminuit.svg?branch=develop
   :target: https://travis-ci.com/HDembinski/iminuit
.. image:: https://ci.appveyor.com/api/projects/status/g6vymxvu9ax34e7l?svg=true
   :target: https://ci.appveyor.com/project/HDembinski/iminuit-b4eg8
.. image:: https://img.shields.io/pypi/v/iminuit.svg
   :target: https://pypi.org/project/iminuit
.. image:: https://codecov.io/gh/HDembinski/iminuit/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/HDembinski/iminuit/branch/develop

.. skip-marker-do-not-remove

*iminuit* is a Python interface to the *MINUIT2* C++ package.

It can be used as a general robust function minimization method,
but is most commonly used for likelihood fits of models to data,
and to get model parameter error estimates from likelihood profile analysis.

* Code: https://github.com/scikit-hep/iminuit
* Documentation: http://iminuit.readthedocs.org/
* Gitter: https://gitter.im/HSF/PyHEP
* Mailing list: https://groups.google.com/forum/#!forum/scikit-hep-forum
* PyPI: https://pypi.org/project/iminuit/
* License: *MINUIT2* is LGPL and *iminuit* is MIT
* Citation: https://github.com/iminuit/iminuit/blob/master/CITATION

In a nutshell
-------------

.. code-block:: python

    from iminuit import Minuit

    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(f)

    m.migrad()  # run optimiser
    print(m.values)  # {'x': 2,'y': 3,'z': 4}

    m.hesse()   # run covariance estimator
    print(m.errors)  # {'x': 1,'y': 1,'z': 1}
