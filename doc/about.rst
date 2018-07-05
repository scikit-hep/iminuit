.. include:: references.txt

.. _about:

About
=====

What is iminuit?
----------------

Interactive IPython-friendly mimizer based on `SEAL Minuit`_.

(It's included in the package, no need to install it separately.)

iminuit is designed from ground up to be fast, interactive and cython friendly. iminuit
extract function signature very permissively starting from checking *func_code*
down to last resort of parsing docstring (or you could tell iminuit to stop looking
and take your answer). The interface is inspired heavily
by PyMinuit and the status printout is inspired by ROOT Minuit. iminuit is
mostly compatible with PyMinuit (with few exceptions). Existing PyMinuit
code can be ported to iminuit by just changing the import statement.

If you are interested in fitting a curve or distribution, take a look at `probfit`_.


Technical Stuff
---------------

Using it as a black box is a bad idea. Here are some fun reads; the order is given
by the order I think you should read.

* Wikipedia for `Quasi Newton Method`_ and `DFP formula`_. The magic behind MIGRAD.
* `Variable Metric Method for Minimization`_ William Davidon 1991
* `A New Approach to Variable Metric Algorithm`_ (R.Fletcher 1970)
* Original Paper: `MINUIT - A SYSTEM FOR FUNCTION MINIMIZATION AND ANALYSIS OF THE PARAMETER ERRORS AND CORRELATIONS`_ by Fred James and Matts Roos.

Team
----

iminuit was created by **Piti Ongmongkolkul** (@piti118). It is a logical successor of pyminuit/pyminuit2, created by **Jim Pivarski** (@jpivarski).

Maintainers
~~~~~~~~~~~

* Piti Ongmongkolkul (@piti118)
* Chih-hsiang Cheng (@gitcheng)
* Christoph Deil (@cdeil)
* Hans Dembinski (@HDembinski)

Contributors
~~~~~~~~~~~~

* Jim Pivarski (@jpivarski)
* David Men\'endez Hurtado (@Dapid)
* Chris Burr (@chrisburr)
* Andrew ZP Smith (@energynumbers)
* Fabian Rost (@fabianrost84)
* Alex Pearce (@alexpearce)
* Lukas Geiger (@lgeiger)
* Omar Zapata (@omazapa)
