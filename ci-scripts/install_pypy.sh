if [ -e $HOME/pypy ]; then
  source $HOME/pypy/venv/bin/activate
else
  mkdir $HOME/pypy
  wget -O - https://bitbucket.org/squeaky/portable-pypy/downloads/pypy3.5-7.0.0-linux_x86_64-portable.tar.bz2 | tar xjf - --strip-components=1 -C $HOME/pypy
  $HOME/pypy/bin/virtualenv-pypy $HOME/pypy/venv --clear
  source $HOME/pypy/venv/bin/activate
  pip install --upgrade cython numpy pytest matplotlib
fi
