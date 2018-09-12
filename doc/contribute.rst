.. include:: references.txt

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

To hack on `iminuit`, start by cloning the repository from `Github`_:

.. code-block:: bash

    $ git clone https://github.com/iminuit/iminuit.git
    $ cd iminuit

Hack away. It is a good idea to develop your feature in a separate branch, so that your master branch remains clean and can follow our master branch.

.. code-block:: bash

    $ git checkout -b "my_cool_feature"
    # now you in a feature branch, commit your edits here

conda
+++++

We recommend you make a dedicated environment for `iminuit`
development, separate from the Python installation you use
for other projects.

One way is to use `conda`_ environments and to use `environment-dev.yml`_
to make the environment and install everything:

.. code-block:: bash

    $ conda env create -f environment-dev.yml
    $ conda activate iminuit-dev

If you ever need to update the environment, you can use:

.. code-block:: bash

    $ conda env update -f environment-dev.yml

It's also easy to deactivate or delete it:

.. code-block:: bash

    $ conda deactivate
    $ conda env remove -n iminuit-dev

virualenv and pip
+++++++++++++++++

Another way is to use `Python virtual environments`_ and to `pip` install via `requirements.txt`_

.. code-block:: bash

   $ python -m venv iminuit-dev
   $ source activate iminuit-dev
   $ pip install -f requirements.txt

Development workflow
--------------------

Hacking on `iminuit` usually means that you edit the Python or Cython files,
and then run a `python setup.py` or `make` command to build the software
or HTML documentation, or to run tests.

The most thing to remember is that we have a `Makefile`_ and the you can run
this command to get help printed concerning the most common available
`make` and `python setup.py` commands:

.. code-block:: bash

   $ make help

Build Minuit and `iminuit` in-place:

.. code-block:: bash

   $ make build

Run the tests:

.. code-block:: bash

    $ make test

Run the notebook tests:

.. code-block:: bash

    $ make test-notebooks

Run the tests with coverage report:

.. code-block:: bash

    $ make coverage
    $ open htmlcov/index.htm

Build the docs:

.. code-block:: bash

   $ make doc
   $ open doc/_build/html/index.html

To check your `iminuit` version number and install location:

.. code-block:: bash

    $ python
    >>> import iminuit
    >>> iminuit
    # install location is printed
    >>> iminuit.__version__
    # version number is printed

.. _requirements.txt: https://github.com/iminuit/iminuit/blob/master/requirements.txt
.. _Python virtual environments: http://docs.python-guide.org/en/latest/dev/virtualenvs/
.. _Github: https://github.com/iminuit/iminuit
.. _conda: https://conda.io/
.. _environment-dev.yml: https://github.com/iminuit/iminuit/blob/master/environment-dev.yml
.. _Makefile: https://github.com/iminuit/iminuit/blob/master/Makefile