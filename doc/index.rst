iminuit
========

Interactive IPython friendly minimizer based on`lcg-minuit`_ .
The popular most minimizer used in High Energy
Physics which has been around for more than 40 years(for a good reason).

It's designed from ground up to be used in an interactive fitting environment
taking full advantage of `IPython notebook <ipythonnb>`_ and Python introspection.
The package is heavily influenced by `PyMinuit`_ and ROOT version of MINUIT.
To make transition from the popular `PyMinuit`_ easy, RTMinuit is (mostly)
compatible with PyMinuit (just change the import statement).

Very Basic Usage. See `tutorial <http://nbviewer.ipython.org/urls/raw.github.com/piti118/iminuit/master/tutorial/tutorial.ipynb>`_ for more comprehensive one::

    def f(x,y,z):
        return (x-1.)**2 + (y-2.)**2 + (z-3.)**2 -1.
    #all parameter constraints are optional
    m=Minuit(f, x=2, error_x=0.2, limit_x=(-10.,10.),
            y=3., fix_y=True,
            print_level=1)
    m.migrad()

    print m.values
    #{'y': 3.0, 'x': 1.0014335661136275, 'z': 3.000257377796712}

    print m.errors
    #{'y': 1.0, 'x': 0.9983245142197408, 'z': 0.9999999998503251}

.. _lcg-minuit: http://seal.web.cern.ch/seal/work-packages/mathlibs/minuit/
.. _PyMinuit: http://code.google.com/p/pyminuit/
.. _ipythonnb: http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html


.. toctree::
    :maxdepth: 2
    :hidden:

    api.rst

Download and Install
--------------------

::

    pip install iminuit

You can find our repository at `github <https://github.com/iminuit/iminuit>`_.

::

    git clone git://github.com/iminuit/iminuit.git

Tutorial
--------

The tutorial is in tutorial directory. You can view it online
`here <http://nbviewer.ipython.org/urls/raw.github.com/piti118/iminuit/master/tutorial/tutorial.ipynb>`_.

API
---

See :ref:`api-doc`

