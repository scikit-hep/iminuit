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

To hack on ``iminuit``, start by cloning the repository from `Github`_:

.. code-block:: bash

    git clone --recursive https://github.com/scikit-hep/iminuit.git
    cd iminuit

It is a good idea to develop your feature in a separate branch, so that your develop branch remains clean and can follow our develop branch.

.. code-block:: bash

    git checkout -b my_cool_feature develop

Now you are in a feature branch, commit your edits here.

Development workflow
--------------------

You have the source code now, next you want to build and test. We recommend to use ``nox``, which automatically creates a dedicated virtual environment for ``iminuit``, separate from the Python installation you use for other projects.

.. code-block:: bash

    nox -s test

This installs your version of ``iminuit`` locally and all the dependencies needed to run the tests, and then runs the tests.

Generate a coverage report:

.. code-block:: bash

    nox -s cov
    <your-web-browser> build/htmlcov/index.htm

If you change something on the C++ side, then ``nox`` is not ideal. In this case you should install the build dependencies ``build``, ``pybind11``, ``scikit-build-core``.

Then you can run

.. code-block:: bash

    python -m build --no-isolation

Note, however, that this only builds the package and does not install it.

If you need to debug the build, you can increase output by adding

.. code-block:: toml

    [tool.scikit-build]
    build.verbose = true

in ``pyproject.toml``. To debug crashes or other issues in the C++ code, you need to compile the wheel with debugging symbols. Add this to the ``pyproject.toml``

.. code-block:: toml

    [tool.scikit-build]
    cmake.build-type = "DEBUG"

compile again and install a debugger. On Linux and MacOS, you can use the GNU debugger

.. code-block:: bash

    gdb --args python -m pytest

On Windows, you can use the wsl shell.

Build the documentation
-----------------------

The documentation is automatically build by the CI/CD pipeline, but if there are issues, you need to build the documentation locally to debug them. You can do that with ``nox``.

.. code-block:: bash

   nox -s doc
   <your-web-browser> build/html/index.html

Making a release
----------------

Maintainers that prepare a release, should follow the instructions in `doc/README.md`

Misc
----

Check your ``iminuit`` version number and install location:

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
