make clean
python setup.py sdist
pip uninstall -y cython
cd dist
pip install iminuit-*.tar.gz
python -c 'import iminuit; iminuit.test()'
