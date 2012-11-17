.. RTMinuit documentation master file, created by
   sphinx-quickstart on Tue Nov 13 10:50:10 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RTMinuit
========

RTMinuit is a python (espcially ) friendly
wrapper of `lcg-minuit`_ . The popular most minimizer used in High Energy
Physics which has been around for more than 40 years(for a good reason).

It's designed from ground up to be used in an interactive fitting environment
taking full advantage of `IPython notebook <ipythonnb>`_.
To make transition from the popular `PyMinuit`_ easy, RTMinuit is (mostly)
compatible with PyMinuit (just change the import statement).

.. _lcg-minuit: http://seal.web.cern.ch/seal/work-packages/mathlibs/minuit/
.. _PyMinuit: http://code.google.com/p/pyminuit/
.. _ipythonnb: http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html

.. toctree::
    :maxdepth: 2
    :hidden:

    api.rst

Download
========

::

    git clone git://github.com/piti118/RTMinuit.git

Prerequisite
============

* `lcg-minuit`_. You can also use patched version from `PyMinuit`_.
  Both will works just fine. This means if you have PyMinuit installed
  you do not need to reinstall Minuit.

Install
=======

::

    python setup.py install


Tutorial
========

The tutorial is in tutorial directory. You can view it online 
:download:`here <tutorial.html>`.

API
===

See :ref:`api-doc`

