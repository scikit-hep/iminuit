# Makefile with some convenient quick ways to do common things

PROJECT = iminuit
PYTHON ?= python

default_target: build

help:
	@echo ''
	@echo ' iminuit available make targets:'
	@echo ''
	@echo '     help             Print this help message'
	@echo ''
	@echo '     build            Build inplace (the default)'
	@echo '     clean            Remove all generated files'
	@echo '     test             Run tests'
	@echo '     test-notebooks   Run notebook tests'
	@echo '     cov              Run tests and write coverage report'
	@echo '     doc              Run Sphinx to generate HTML docs'
	@echo ''
	@echo '     integration      Run integration check'
	@echo '     release          Prepare a release (for maintainers)'
	@echo ''
	@echo '     code-analysis    Run code analysis (flake8 and pylint)'
	@echo '     flake8           Run code analysis (flake8)'
	@echo '     pylint           Run code analysis (pylint)'
	@echo ''
	@echo ' More info:'
	@echo ''
	@echo ' * iminuit code: https://github.com/scikit-hep/iminuit'
	@echo ' * iminuit docs: https://iminuit.readthedocs.org/'
	@echo ''

clean:
	rm -rf build htmlcov doc/_build iminuit/_libiminuit.cpp iminuit/_libiminuit*.so tutorial/.ipynb_checkpoints iminuit.egg-info .pytest_cache iminuit/__pycache__ iminuit/tests/__pycache__

build: iminuit/_libiminuit.so

iminuit/_libiminuit.so: $(wildcard Minuit/src/*.cxx iminuit/*.pyx iminuit/*.pxi)
	$(PYTHON) setup.py build_ext --inplace

test: build
	$(PYTHON) -m pytest iminuit

test-notebooks: build
	$(PYTHON) test_notebooks.py

cov: build
	@echo "Note: This only shows the coverage in pure Python."
	$(PYTHON) -m pytest iminuit --cov iminuit --cov-report html

doc/_build/html/index.html: iminuit/_libiminuit.so $(wildcard doc/*.rst)
	{ cd doc; make html; }

doc: doc/_build/html/index.html

code-analysis: flake8 pylint

flake8:
	$(PYTHON) -m flake8 --max-line-length=90 $(PROJECT) | grep -v __init__ | grep -v external

# TODO: once the errors are fixed, remove the -E option and tackle the warnings
pylint:
	$(PYTHON) -m pylint -E $(PROJECT)/ -d E1103,E0611,E1101 \
	       --ignore="" -f colorized \
	       --msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'

conda:
	$(PYTHON) setup.py bdist_conda

sdist:
	rm -rf dist iminuit.egg-info
	$(PYTHON) setup.py sdist

integration:
	@echo
	@echo "Warning: If integration tests fail, add new tests of corrupted interface to iminuit."
	@echo
	.ci/gammapy_integration_test.sh && .ci/probfit_integration_test.sh

release: sdist
	pip install --upgrade twine
	@echo ""
	@echo "Release checklist:"
	@echo "[ ] Integration tests ok 'make integration'"
	@echo "[ ] Increase version number in iminuit/info.py"
	@echo "[ ] Update doc/changelog.rst"
	@echo "[ ] Tag release on Github"
	@echo ""
	@echo "Upload to TestPyPI:"
	@echo "twine upload --repository-url https://test.pypi.org/legacy/ dist/*"
	@echo ""
	@echo "Upload to PyPI:"
	@echo "twine upload dist/*"
	@echo ""