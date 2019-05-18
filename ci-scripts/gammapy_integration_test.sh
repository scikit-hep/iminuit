#!/bin/bash
# run this before a release to check that gammapy is not broken
if [ ! -e gammapy ]; then
  virtualenv gammapy
fi
source gammapy/bin/activate || { echo "Error: You must execute or source this script!"; exit 0; }
pip install cython numpy pytest
pip install .
pip install gammapy pytest-astropy
cd /tmp
gammapy info
python -c "import sys, gammapy; sys.exit(gammapy.test())"
