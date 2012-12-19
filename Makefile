# simple makefile to simplify repetetive build env management tasks under posix

PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-pyc:
	find iminuit -name "*.pyc" | xargs rm -f

clean-so:
	find iminuit -name "*.so" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

install:
	$(PYTHON) setup.py install

install-user:
	$(PYTHON) setup.py install --user

sdist: clean
	$(PYTHON) setup.py sdist

register:
	$(PYTHON) setup.py register

upload: clean
	$(PYTHON) setup.py sdist upload

test-code: in
	$(NOSETESTS) -s iminuit

test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture doc/

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=iminuit iminuit

test: test-code test-doc

trailing-spaces:
	find iminuit -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

doc: inplace
	make -C doc/ html

cython:
	cython -a --cplus --fast-fail --line-directives iminuit/_libiminuit.pyx

check-rst:
	python setup.py --long-description | rst2html.py > __output.html
	rm -f __output.html
