#!/bin/bash
python3 .ci/install_pypy36.py
source pypy36/pypy3/bin/activate || { echo "Error: You must execute or source this script!"; exit 1; }
pip install --upgrade cython numpy==1.15.4 pytest
