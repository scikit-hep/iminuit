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
	@echo '     cov              Run tests and write coverage report'
	@echo '     doc              Run Sphinx to generate HTML docs'
	@echo ''
	@echo '     integration      Run integration check'
	@echo '     release          Prepare a release (for maintainers)'
	@echo ''
	@echo '     analysis         Run code analysis (flake8 and pylint)'
	@echo '     flake8           Run code analysis (flake8)'
	@echo '     pylint           Run code analysis (pylint)'
	@echo ''
	@echo ' More info:'
	@echo ''
	@echo ' * iminuit code: https://github.com/scikit-hep/iminuit'
	@echo ' * iminuit docs: https://iminuit.readthedocs.org/'
	@echo ''

clean:
	rm -rf build htmlcov doc/_build src/iminuit/_libiminuit.cpp src/iminuit/_libiminuit*.so tutorial/.ipynb_checkpoints iminuit.egg-info .pytest_cache src/iminuit/__pycache__ src/iminuit/tests/__pycache__ tutorial/__pycache__ .coverage .eggs .ipynb_checkpoints dist

build: src/iminuit/_libiminuit.so

src/iminuit/_libiminuit.so: $(wildcard src/iminuit/*.pyx src/iminuit/*.pxi)
	$(PYTHON) setup.py build_ext --inplace

test: build
	$(PYTHON) -m pytest

cov: build
	@echo "Note: This only shows the coverage in pure Python."
	$(PYTHON) -m pytest --cov src/iminuit --cov-report html

doc/_build/html/index.html: build $(wildcard src/iminuit/*.pyx src/iminuit/*.pxi src/iminuit/*.py src/iminuit/**/*.py doc/*.rst)
	{ cd doc; make html; }

doc: doc/_build/html/index.html

analysis: flake8 pylint

flake8: build
	@$(PYTHON) -m flake8 --extend-ignore=E203 --max-line-length=95 src/$(PROJECT)

# TODO: once the errors are fixed, remove the -E option and tackle the warnings
pylint: build
	@$(PYTHON) -m pylint -E src/$(PROJECT) -d E0611,E0103,E1126 -f colorized \
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
	.ci/probfit_integration_test.sh
