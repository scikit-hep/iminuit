iminuit
========

Interactive IPython Friendly Mimizer based on `SEAL Minuit <http://seal.web.cern.ch/seal/work-packages/mathlibs/minuit/>`_.
(It's included in the package no need to install it separately)

It is designed from ground up to be fast, interactive and cython friendly. iminuit
extract function signature very permissively starting from checking *func_code*
down to last resort of parsing docstring(or you could tell iminuit to stop looking
and take your answer). The interface is inspired heavily
by PyMinuit and the status printout is inspired by ROOT Minuit. iminuit is
mostly compatible with PyMinuit(with few exceptions). Existing PyMinuit
code can be ported to iminuit by just changing the import statement.

In a nutshell,::

    from iminuit import Minuit
    def f(x,y,z):
        return (x-2)**2 + (y-3)**2 + (z-4)**2
    m = Minuit(f)
    m.migrads()
    print m.values #{'x':2,'y':3,'z':4}
    print m.errors

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

Tutorial
--------

All the tutorials are in tutorial directory. You can view it online too.

- `Quick start <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/iminuit/master/tutorial/tutorial.ipynb>`_
- `Hard Core Cython tutorial <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/iminuit/master/tutorial/hard-core-tutorial.ipynb>`_.
  If you need to do a huge likelihood fit that need speed.
  This is for you. If you don't care, just use dist_fit. It's a fun
  read though I think.


API
---

See :ref:`api-doc`

