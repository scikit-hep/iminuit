# Makefile for developers with some convenient quick ways to do common things

ifeq ($(shell python -c 'import sys; print(sys.version_info.major)'), 3)
	PYTHON = python
else
	PYTHON = python3
endif

# default target
build/done: $(wildcard *.py src/*.cpp extern/root/math/minuit2/src/*.cxx extern/root/math/minuit2/inc/*.h) CMakeLists.txt
	$(PYTHON) setup.py develop
	touch build/done

test: build/done
	$(PYTHON) -m pytest -vv -r a --ff --pdb

cov: build/done
	@echo "Note: This only shows the coverage in pure Python."
	rm -rf htmlcov
	$(PYTHON) -m pytest -x --ff --cov src/iminuit --cov-report html
	xdg-open htmlcov/index.html

doc: build/done build/html/done
	xdg-open build/html/index.html

build/html/done: doc/conf.py $(wildcard src/iminuit/*.py doc/*.rst doc/_static/* doc/plots/*)
	$(PYTHON) -c "import iminuit" # requires iminuit to be installed
	mkdir -p build/html
	sphinx-build -W -a -E -b html -d build/doctrees doc build/html
	touch build/html/done

check:
	pre-commit run -a

clean:
	rm -rf build htmlcov src/iminuit/_core* tutorial/.ipynb_checkpoints iminuit.egg-info src/iminuit.egg-info .pytest_cache src/iminuit/__pycache__ tests/__pycache__ tutorial/__pycache__ .coverage .eggs .ipynb_checkpoints dist __pycache__

.PHONY: clean check cov doc

## pylint is garbage, also see https://lukeplant.me.uk/blog/posts/pylint-false-positives/#running-total
# pylint: build
# 	@$(PYTHON) -m pylint -E src/$(PROJECT) -d E0611,E0103,E1126 -f colorized \
# 	       --msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'
