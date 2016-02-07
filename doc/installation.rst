.. _installation:

Installation
============

iminuit works with Python 2.7 as well as 3.4 or later.

Dependencies
------------

Like most Python packages, iminuit installation requires `setuptools <https://pypi.python.org/pypi/setuptools>`__

The following dependencies are optional:

* numpy
* ipython
* matplotlib
* pytest, pytest-cov
* Cython
* Sphinx, sphinx-rtd


TODO: describe better where which dependency is used.


Stable version
--------------

To install the latest stable version:

.. code-block:: bash

    $ pip install iminuit


Conda
-----

Conda packages for iminuit are available via the astropy channel at https://anaconda.org/astropy/iminuit

.. code-block:: bash

    $ conda install -c astropy iminuit


Windows
-------

For Windows, Christoph Gohlke made a nice windows binary to save you all from Windows compilation nightmare:

   `http://www.lfd.uci.edu/~gohlke/pythonlibs/#iminuit <http://www.lfd.uci.edu/~gohlke/pythonlibs/#iminuit>`_


Development version
-------------------

To install the latest development version clone the
repository from `Github <https://github.com/iminuit/iminuit>`_:

.. code-block:: bash

    $ git clone git://github.com/iminuit/iminuit.git
    $ cd iminuit
    $ python setup.py install

Docs
----

To generate html docs locally:

.. code-block:: bash

   $ python setup.py build_ext --inplace
   $ cd doc
   $ make html
   $ open _build/html/index.html

You will need ``sphinx`` and ``sphinx_rtd_theme``.
They can be installed via

.. code-block:: bash

   $ pip install sphinx
   $ pip install sphinx_rtd_theme

Testing
-------

To run the tests you need to install `pytest <http://pytest.org>`_.

To run the iminuit tests for an installed version of the package:

.. code-block:: bash

    python -m pytest --pyargs iminuit

To run the tests from the source folder (e.g. during pytest development), use these commands:

.. code-block:: bash

       $ python setup.py build_ext --inplace
       $ python -m pytest -v iminuit
       $ python -m pytest -v iminuit --cov iminuit --cov-report html
