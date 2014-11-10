.. _installation:

Installation
============

Stable version
--------------

To install the latest stable version:

.. code-block:: bash

    $ pip install iminuit

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

To run the tests:

.. code-block:: bash

   $ python setup.py build_ext --inplace
   $ nosetests -V

You will need ``nose``.
It can be installed via

.. code-block:: bash

   $ pip install nose
