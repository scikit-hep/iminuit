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

    git clone https://github.com/scikit-hep/iminuit.git
    cd iminuit

It is a good idea to develop your feature in a separate branch, so that your develop branch remains clean and can follow our develop branch.

.. code-block:: bash

    git checkout -b my_cool_feature develop

Now you are in a feature branch, commit your edits here.

virtualenv
++++++++++

You have the source code now, but you also want to build and test. We recommend to make a dedicated build environment for `iminuit`, separate from the Python installation you use for other projects.

One way is to use `Python virtual environments`_ and `pip` to install the development packages listed in `requirements-dev.txt`_

.. code-block:: bash

    pip install virtualenv
    virtualenv iminuit-dev
    source iminuit-dev/bin/activate
    pip install -r requirements-dev.txt

To delete the virtual environment just delete the folder iminuit-dev.

conda
+++++

Another way is to use `conda`_ environments and `environment-dev.yml`_
to make the environment and install everything. You need to install a conda first, e.g. `miniconda`_:

.. code-block:: bash

    conda env create -f environment-dev.yml
    conda activate iminuit-dev

If you ever need to update the environment, you can use:

.. code-block:: bash

    conda env update -f environment-dev.yml

It's also easy to deactivate or delete it:

.. code-block:: bash

    conda deactivate
    conda env remove -n iminuit-dev

Development workflow
--------------------

To simplify hacking, we have a Makefile with common commands. To see what commands are available, do:

.. code-block:: bash

   make help

Build `iminuit` in-place:

.. code-block:: bash

   make

Run the tests:

.. code-block:: bash

    make test

Run the notebook tests:

.. code-block:: bash

    make test-notebooks

Run the tests and generate a coverage report:

.. code-block:: bash

    make cov
    <your-web-browser> htmlcov/index.htm

Build the docs:

.. code-block:: bash

   make doc
   <your-web-browser> doc/_build/html/index.html

If you change the public interface of iminuit, you should run the integration tests in addition to iminuits internal tests. The integration tests will download and install consumers of iminuit to check their tests. This allows us to see that iminuit does not break them.

.. code-block:: bash

    make integration

Ideally, the integration tests should never fail because of iminuit, because breaking changes in the public interface should be detected by our own unit tests. If you find a problem during integration, *you should add new tests to iminuit* which will detect this problem in the future without relying on others!

Maintainers that prepare a release, should run:

.. code-block:: bash

    make release

It generates the source distribution and prints a checklist for the release.

To check your `iminuit` version number and install location:

.. code-block:: bash

    $ python
    >>> import iminuit
    >>> iminuit
    # install location is printed
    >>> iminuit.__version__
    # version number is printed

.. _conda: https://conda.io/
.. _miniconda: https://conda.io/en/latest/miniconda.html
.. _environment-dev.yml: https://github.com/scikit-hep/iminuit/blob/master/environment-dev.yml
.. _requirements-dev.txt: https://github.com/scikit-hep/iminuit/blob/master/requirements-dev.txt
.. _Python virtual environments: http://docs.python-guide.org/en/latest/dev/virtualenvs/
.. _Github: https://github.com/scikit-hep/iminuit
.. _Makefile: https://github.com/scikit-hep/iminuit/blob/master/Makefile
