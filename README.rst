iminuit
=======

**Python interface to the MINUIT2 C++ package - Fitting like a boss!**

======  ======================  ========================
Branch  Linux (Py-2.7, Py-3.6)  Windows (Py-2.7, Py-3.6)
======  ======================  ========================
master  [![Build Status Travis](https://travis-ci.org/iminuit/iminuit.svg?branch=master)](https://travis-ci.org/iminuit/iminuit?branch=master)  [![Build status Appveyor](https://ci.appveyor.com/api/projects/status/g6vymxvu9ax34e7l?svg=true)](https://ci.appveyor.com/project/HDembinski/iminuit-b4eg8)
======  ======================  ========================

iminuit works with Python-2.7 to 3.5 on Windows, Mac (currently manually tested), and Linux.

It can be used as a general robust function minimisation method, but it really
shines in statistical likelihood fits of models to data. Use iminuit to get
uncertainty estimates of model parameters with the profile likelihood method.

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
