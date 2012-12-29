.. -*- mode: rst -*-

.. image:: https://travis-ci.org/iminuit/iminuit.png?branch=master
   :target: https://travis-ci.org/iminuit/iminuit

iminuit
--------

Interactive IPython Friendly Mimizer based on `SEAL Minuit <http://seal.web.cern.ch/seal/work-packages/mathlibs/minuit/release/download.html>`_.
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
    m.migrad()
    print m.values #{'x':2,'y':3,'z':4}
    print m.errors

Install
-------

::

    python setup.py install

or from pip::

    pip install iminuit

Tutorial
--------

All the tutorials are in tutorial directory. You can view it online too.

- `Quick start <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/iminuit/master/tutorial/tutorial.ipynb>`_
- `Hard Core Cython tutorial <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/iminuit/master/tutorial/hard-core-tutorial.ipynb>`_.
  If you need to do a huge likelihood fit that need speed or learn how to
  parallelize your stuff, this is for you.


Documentation
-------------

http://iminuit.github.com/iminuit/

Technical Stuff
---------------

Using it as a black box is a bad idea. Here are some fun read the order is given
by the order I think you should read.

Wikipedia for `Quasi Newton Method <http://en.wikipedia.org/wiki/Quasi-Newton_method>`_ and
`DFP formula <http://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula>`_.
The magic behind migrad.

`Variable Metric Method for Minimization <http://www.ii.uib.no/~lennart/drgrad/Davidon1991.pdf>`_ William Davidon 1991

`A New Approach to Variable Metric Algorithm. <http://comjnl.oxfordjournals.org/content/13/3/317.full.pdf+html>`_ (R.Fletcher 1970)

Original Paper: `MINUIT - A SYSTEM FOR FUNCTION MINIMIZATION AND ANALYSIS OF THE PARAMETER ERRORS AND CORRELATIONS <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.158.9157&rep=rep1&type=pdf>`_ by Fred James and Matts Roos.

You can help
------------

Github allows you to contribute to this project very easily just fork the
repository, make changes and submit a pull request.

Here is some areas you can help.

- Documentation. Tell us what's missing, what's incorrect or misleading.
  Look at doc directory and gh-pages branch. I'm the one who wrote the package,
  so I know how it works in detail. I might have skipped something important
  in documentation.
- HTML output looks ugly. Yah.. I'm lazy. Please help me change it.
- Test test test. This package is realtively your there might be some kinks to
  it let me know how you broke it.
- Performance. I you are a C/cython/python hacker go ahead and make it faster.
- Console output could use some color.

