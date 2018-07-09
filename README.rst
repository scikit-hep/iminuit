iminuit
=======

MINUIT from Python - Fitting like a boss

`iminuit` is a Python interface to the `MINUIT` C++ package.

It can be used as a general robust function minimisation method,
but is most commonly used for likelihood fits of models to data,
and to get model parameter error estimates from likelihood profile analysis.

* Code: https://github.com/iminuit/iminuit
* Documentation: http://iminuit.readthedocs.org/
* Mailing list: https://groups.google.com/forum/#!forum/iminuit
* PyPI: https://pypi.org/project/iminuit/
* License: MINUIT is LGPL and iminuit is MIT
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
