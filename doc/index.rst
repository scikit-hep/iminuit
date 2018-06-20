.. include:: references.txt

iminuit
=======

MINUIT from Python - Fitting like a boss

* Code: https://github.com/iminuit/iminuit
* Documentation: http://iminuit.readthedocs.org/
* Mailing list: https://groups.google.com/forum/#!forum/iminuit
* PyPI: https://pypi.python.org/pypi/iminuit
* License: LGPL (the iminuit source is MIT, but the bundled MINUIT is LGPL and thus the whole package is LGPL)
* Citation: https://github.com/iminuit/iminuit/blob/master/CITATION


In a nutshell
-------------

.. code-block:: python

    from iminuit import Minuit
    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2
    m = Minuit(f)
    m.migrad()
    print(m.values)  # {'x': 2,'y': 3,'z': 4}
    print(m.errors)  # {'x': 1,'y': 1,'z': 1}

If you are interested in fitting a curve or distribution, take a look at `probfit`_.


.. toctree::
    :maxdepth: 4
    :hidden:

    install
    about
    api
    contribute


Tutorial
--------

All the tutorials are in tutorial directory. You can view them online too:

- `Quick start <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/iminuit/master/tutorial/tutorial.ipynb>`_
- `Hard Core Cython tutorial <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/iminuit/master/tutorial/hard-core-tutorial.ipynb>`_.
  If you need to do a huge likelihood fit that needs speed, this is for you.
  If you don't care, just use `probfit`_.
  It's a fun read though I think.

API
---

See :ref:`api-doc`
