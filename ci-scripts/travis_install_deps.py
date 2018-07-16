import pip
import os
import subprocess as subp


def install(package_string):
    pip.main(['install'] + package_string.split())


build = os.environ['BUILD'].lower()
if build == 'conda':
    subp.call_check(['sh', 'install_conda.sh'])
elif build == 'all':
    install('cython numpy pytest matplotlib scipy ipython sphinx sphinx_rtd_theme jupyter')
elif build == 'coverage':
    install('cython numpy pytest matplotlib scipy ipython sphinx sphinx_rtd_theme jupyter pytest-cov')
elif build == 'sdist':
    install('cython numpy pytest matplotlib scipy ipython')
elif build == 'minimal':
    install('cython numpy pytest')
else:
    raise ValueError('build option not recognized')

# https://docs.travis-ci.com/user/multi-os/
# This might also be useful:
# https://stackoverflow.com/questions/45257534/how-can-i-build-a-python-project-with-osx-environment-on-travis
#  - if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then brew update          ; fi
#  - if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then brew install python  ; fi
