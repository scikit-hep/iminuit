time python setup.py build_ext -i
time nosetests

if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then
  time python setup.py install
  cd tutorial
  time ../devscripts/checkipnb.py tutorial.ipynb
  time ../devscripts/checkipnb.py hard-core-tutorial.ipynb
fi
