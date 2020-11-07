# Makefile with some convenient quick ways to do common things

ifeq ($(shell python -c 'import sys; print(sys.version_info.major)'), 3)
	PYTHON = python
else
	PYTHON = python3
endif

default_target: build

help:
	@echo ''
	@echo ' iminuit available make targets:'
	@echo ''
	@echo '     help             Print this help message'
	@echo ''
	@echo '     build            Build inplace (the default)'
	@echo '     clean            Remove all generated files'
	@echo '     test             Run tests (excluding tutorials)'
	@echo '     cov              Run tests and write coverage report'
	@echo '     doc              Run Sphinx to generate HTML docs'
	@echo ''
	@echo ' More info:'
	@echo ''
	@echo ' * iminuit code: https://github.com/scikit-hep/iminuit'
	@echo ' * iminuit docs: https://iminuit.readthedocs.org/'
	@echo ''

clean:
	rm -rf build htmlcov doc/_build src/iminuit/_core*.so tutorial/.ipynb_checkpoints iminuit.egg-info src/iminuit.egg-info .pytest_cache src/iminuit/__pycache__ src/iminuit/tests/__pycache__ tutorial/__pycache__ .coverage .eggs .ipynb_checkpoints dist

build: build/log

test: build
	$(PYTHON) -m pytest -qvv --pdb

cov: build
	@echo "Note: This only shows the coverage in pure Python."
	$(PYTHON) -m pytest --cov src/iminuit --cov-report html

doc: doc/_build/html/index.html

build/log: $(wildcard src/*.cpp) $(wildcard extern/root/math/minuit2/src/*.cxx) $(wildcard extern/root/math/minuit2/inc/*.h) CMakeLists.txt setup.py cmake.py
	$(PYTHON) setup.py build_ext --inplace -g
	touch build/log

doc/_build/html/index.html: build $(wildcard src/*.cpp src/iminuit/*.py src/iminuit/**/*.py doc/*.rst)
	{ cd doc; make html; }

## pylint is garbage, also see https://lukeplant.me.uk/blog/posts/pylint-false-positives/#running-total
# pylint: build
# 	@$(PYTHON) -m pylint -E src/$(PROJECT) -d E0611,E0103,E1126 -f colorized \
# 	       --msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'
