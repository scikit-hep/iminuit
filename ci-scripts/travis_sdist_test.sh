make clean
make sdist
virtualenv dist
source dist/bin/activate
cd dist
IMINUIT=$( ls iminuit-*.tar* )
pip install ${IMINUIT}[tests]
python -c 'import iminuit; iminuit.test()'
