# Makefile for developers with some convenient quick ways to do common things

# default target
build/done: $(wildcard *.py src/*.cpp extern/root/math/minuit2/src/*.cxx extern/root/math/minuit2/inc/*.h) CMakeLists.txt
	mkdir -p build
	python .ci/install_deps.py build
	DEBUG=1 CMAKE_BUILD_PARALLEL_LEVEL=8 CMAKE_ARGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache" python -m pip install --no-build-isolation -v -e .
	touch build/done

build/testdep: build/done
	python .ci/install_deps.py test
	touch build/testdep

test: build/done build/testdep
	JUPYTER_PLATFORM_DIRS=1 python -m pytest -vv -r a --ff --pdb

# this requires that ROOT is installed
bench: build/done build/testdep
	python -m pytest bench --benchmark-autosave

# only computes the coverage in pure Python
cov: build/done build/testdep
	rm -rf htmlcov
	JUPYTER_PLATFORM_DIRS=1 coverage run -m pytest
	coverage html -d htmlcov
	coverage report -m
	@echo htmlcov/index.html

doc: build/done build/html/done
	@echo build/html/index.html

build/docdep: build/done
	python .ci/install_deps.py test doc
	touch build/docdep

build/html/done: build/done build/docdep doc/conf.py $(wildcard src/iminuit/*.py doc/*.rst doc/_static/* doc/plots/* doc/notebooks/*.ipynb doc/notebooks/roofit/*.ipynb *.rst)
	mkdir -p build/html
	sphinx-build -v -W -b html -d build/doctrees doc build/html
	touch build/html/done

tutorial: build/done build/tutorial_done

build/tutorial_done: $(wildcard src/iminuit/*.py doc/tutorial/*.ipynb)
	python3 -m pytest -n8 doc/tutorial
	touch build/tutorial_done

check:
	pre-commit run -a

clean:
	rm -rf build htmlcov src/iminuit/_core* tutorial/.ipynb_checkpoints iminuit.egg-info src/iminuit.egg-info .pytest_cache src/iminuit/__pycache__ tests/__pycache__ tutorial/__pycache__ .coverage .eggs .ipynb_checkpoints dist __pycache__

.PHONY: clean check cov doc test bench
