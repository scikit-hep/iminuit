import sys
import os
import subprocess

build = os.environ['BUILD']


def pip_install(packages):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages.split())


def main():
    if build == 'CONDA':
        subprocess.check_call(['sh', 'ci-scripts/travis_install_conda.sh'])
    elif build == 'ALL':
        pip_install('cython numpy pytest matplotlib scipy ipython sphinx sphinx_rtd_theme jupyter')
    elif build in {'TEST', 'COVERAGE'}:
        pip_install('cython numpy pytest matplotlib scipy ipython sphinx sphinx_rtd_theme jupyter pytest-cov')
    elif build == 'SDIST':
        pip_install('cython numpy pytest matplotlib scipy ipython')
    elif build == 'MINIMAL':
        pip_install('cython numpy pytest')
    else:
        raise ValueError('build option not recognized: {!r}'.format(build))


if __name__ == '__main__':
    main()
