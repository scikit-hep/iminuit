if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then
  # For Python 2.7 we can install binary packages with apt-get
  # We use this to check the notebooks
  deactivate;
  time sudo apt-get install -qq python-nose python-matplotlib ipython-notebook cython;
fi
#- time pip install ipython[zmq,notebook] --use-mirrors
#- time pip install cython --use-mirrors
