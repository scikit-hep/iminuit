#!/bin/bash
if [ ! -e pypy3.6/venv/bin/activate ]; then
  mkdir pypy3.6
  wget -O - https://bitbucket.org/squeaky/portable-pypy/downloads/pypy3.6-7.1.1-beta-linux_x86_64-portable.tar.bz2 | tar xjf - --strip-components=1 -C pypy3.6
  pypy3.6/bin/virtualenv-pypy pypy3.6/venv
fi
  source pypy3.6/venv/bin/activate || { echo "Error: You must execute or source this script!"; exit 1; }
  pip install --upgrade cython numpy==1.15.4 pytest
