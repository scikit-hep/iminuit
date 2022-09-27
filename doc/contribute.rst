.. include:: bibliography.txt

.. _contribute:

Contribute
==========

You can help
------------

Please open issues and feature requests on `Github`_. We respond quickly.

* Documentation. Tell us what's missing, what's incorrect or misleading.
* Tests. If you have an example that shows a bug or problem, please file an issue!
* Performance. If you are a C/cython/python hacker and see a way to make the code faster, let us know!

Direct contributions related to these items are welcome, too! If you want to contribute, please `fork the project on Github <https://help.github.com/articles/fork-a-repo>`_, develop your change and then make a `pull request <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_. This allows us to review and discuss the change with you, which makes the integration very smooth.

Development setup
-----------------

git
+++

To hack on ``iminuit``, start by cloning the repository from `Github`_:

.. code-block:: bash

    git clone --recursive https://github.com/scikit-hep/iminuit.git
    cd iminuit

It is a good idea to develop your feature in a separate branch, so that your develop branch remains clean and can follow our develop branch.

.. code-block:: bash

    git checkout -b my_cool_feature develop

Now you are in a feature branch, commit your edits here.

venv
++++

You have the source code now, but you also want to build and test. We recommend to make a dedicated virtual environment for ``iminuit``, separate from the Python installation you use for other projects.

The recommended way is to use `Python virtual environments`_ and ``pip``.

.. code-block:: bash

    python -m venv py3 # folders called py* are ignored by .gitignore
    source py3/bin/activate
    pip install -e '.[test]'

To delete the virtual environment just delete the folder ``py3``.

Development workflow
--------------------

To simplify hacking on OSX and Linux, we have a Makefile with common commands.

Build ``iminuit`` in-place and run the tests:

.. code-block:: bash

   make test

Same and generate a coverage report:

.. code-block:: bash

    make cov
    <your-web-browser> htmlcov/index.htm

Build the docs:

.. code-block:: bash

   make doc
   <your-web-browser> doc/_build/html/index.html

Run the notebook tests:

.. code-block:: bash

   make tutorial

Maintainers that prepare a release, should follow the instructions in `doc/README.md`_

To check your ``iminuit`` version number and install location:

.. code-block:: bash

    $ python
    >>> import iminuit
    >>> iminuit
    # install location is printed
    >>> iminuit.__version__
    # version number is printed

.. _conda: https://conda.io/
.. _miniconda: https://conda.io/en/latest/miniconda.html
.. _Python virtual environments: http://docs.python-guide.org/en/latest/dev/virtualenvs/
.. _Github: https://github.com/scikit-hep/iminuit
.. _Makefile: https://github.com/scikit-hep/iminuit/blob/main/Makefile
.. _doc/README.md: https://github.com/scikit-hep/iminuit/blob/main/doc/README.md
