#!/bin/bash
# run this before a release to check that gammapy is not broken
if [ ! -e probfit ]; then
  virtualenv probfit
fi
source probfit/bin/activate || { echo "Error: You must execute or source this script!"; exit 1; }
pip install cython numpy pytest pytest-mpl 'matplotlib<=3.0.0'
pip install .
wget -O - https://github.com/scikit-hep/probfit/archive/1.1.0.tar.gz | tar xzf - --strip-components=1 -C probfit
cd probfit
make build
make test
