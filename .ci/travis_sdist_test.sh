#!/bin/bash
make clean
make sdist
virtualenv dist
source dist/bin/activate || { echo "Error: You must execute or source this script!"; exit 1; }
cd dist
IMINUIT=$( ls iminuit-*.tar* )
pip install ${IMINUIT}[tests]
python -c 'import sys, iminuit; sys.exit(iminuit.test())'
