# Makefile with some convenient quick ways to do common things

PROJECT = iminuit
CYTHON ?= cython

default_target: build

help:
	@echo ''
	@echo ' iminuit available make targets:'
	@echo ''
	@echo '     help             Print this help message'
	@echo ''
	@echo '     build            Build inplace (the default)'
	@echo '     clean            Remove generated files'
	@echo '     test             Run tests'
	@echo '     coverage         Run tests and write coverage report'
	@echo '     cython           Compile cython files'
	@echo '     doc              Run Sphinx to generate HTML docs'
	@echo ''
	@echo '     code-analysis    Run code analysis (flake8 and pylint)'
	@echo '     flake8           Run code analysis (flake8)'
	@echo '     pylint           Run code analysis (pylint)'
	@echo ''
	@echo ' Note that most things are done via `python setup.py`, we only use'
	@echo ' make for things that are not trivial to execute via `setup.py`.'
	@echo ''
	@echo ' Common `setup.py` commands:'
	@echo ''
	@echo '     python setup.py --help-commands'
	@echo '     python setup.py install'
	@echo '     python setup.py develop'
	@echo '     python setup.py test -V'
	@echo '     python setup.py test --help # to see available options'
	@echo '     python setup.py build_sphinx # use `-l` for clean build'
	@echo ''
	@echo ' More info:'
	@echo ''
	@echo ' * iminuit code: https://github.com/iminuit/iminuit'
	@echo ' * iminuit docs: https://iminuit.readthedocs.org/'
	@echo ''

clean:
	rm -rf build htmlcov doc/_build iminuit/_libiminuit.cpp
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

build: iminuit/_libiminuit.so

iminuit/_libiminuit.so: $(wildcard Minuit/src/*.cxx Minuit/inc/*/*.h iminuit/*.pyx iminuit/*.pxi)
	python setup.py build_ext --inplace

test: build
	python -m pytest -v iminuit

coverage: build
	python -m pytest -v iminuit --cov iminuit --cov-report html --cov-report term-missing --cov-report xml

# TODO: this runs Cython differently than via setup.py ... make it consistent!
cython:
	find $(PROJECT) -name "*.pyx" -exec $(CYTHON) --cplus  {} \;

doc/_build/html/index.html: iminuit/_libiminuit.so $(wildcard doc/*.rst)
	{ cd doc; make html; }

doc: doc/_build/html/index.html

code-analysis: flake8 pylint

flake8:
	flake8 --max-line-length=90 $(PROJECT) | grep -v __init__ | grep -v external

# TODO: once the errors are fixed, remove the -E option and tackle the warnings
pylint:
	pylint -E $(PROJECT)/ -d E1103,E0611,E1101 \
	       --ignore="" -f colorized \
	       --msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'
