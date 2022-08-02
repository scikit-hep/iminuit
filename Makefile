# Makefile for developers with some convenient quick ways to do common things

# default target
build/done: build/deps $(wildcard *.py src/*.cpp extern/root/math/minuit2/src/*.cxx extern/root/math/minuit2/inc/*.h) CMakeLists.txt
	DEBUG=1 python -m pip install -v -e .
	touch build/done

test: build/done
	python -m pytest -vv -r a --ff --pdb

build/deps:
	# (re-)install all test dependencies
	python .ci/install_extra.py
	touch build/deps

cov: build/done
	# only computes the coverage in pure Python
	rm -rf htmlcov
	coverage run -m pytest
	python -m pip uninstall --yes numba ipykernel ipywidgets
	coverage run --append -m pytest
	python -m pip uninstall --yes scipy matplotlib
	coverage run --append -m pytest
	coverage html -d htmlcov
	rm build/deps

doc: build/done build/html/done

build/html/done: doc/conf.py $(wildcard src/iminuit/*.py doc/*.rst doc/_static/* doc/plots/* doc/tutorial/*.ipynb *.rst)
	mkdir -p build/html
	sphinx-build -j3 -W -a -E -b html -d build/doctrees doc build/html
	touch build/html/done

tutorial: build/done build/tutorial_done

build/tutorial_done: $(wildcard src/iminuit/*.py doc/tutorial/*.ipynb)
	python3 -m pytest -n8 doc/tutorial
	touch build/tutorial_done

check:
	pre-commit run -a

clean:
	rm -rf build htmlcov src/iminuit/_core* tutorial/.ipynb_checkpoints iminuit.egg-info src/iminuit.egg-info .pytest_cache src/iminuit/__pycache__ tests/__pycache__ tutorial/__pycache__ .coverage .eggs .ipynb_checkpoints dist __pycache__

.PHONY: clean check cov doc test

## pylint is garbage, also see https://lukeplant.me.uk/blog/posts/pylint-false-positives/#running-total
# pylint: build
# 	@python3 -m pylint -E src/$(PROJECT) -d E0611,E0103,E1126 -f colorized \
# 	       --msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'
