# -*- coding: utf-8 -*-
import sys
from os.path import dirname, join
from glob import glob
from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand


# http://pytest.org/latest/goodpractices.html#manual-integration
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        # self.pytest_args = '--pyargs iminuit'
        # self.pytest_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.pytest_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


# Static linking
cwd = dirname(__file__)
minuit_src = glob(join(cwd, 'Minuit/src/*.cxx'))
minuit_header = join(cwd, 'Minuit/inc')

# We follow the recommendation how to distribute Cython modules:
# http://docs.cython.org/src/reference/compilation.html#distributing-cython-modules
try:
    from Cython.Build import cythonize

    USE_CYTHON = True  # TODO: add command line option?
except ImportError:
    print('Cython is not available ... using pre-generated C file.')
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.cpp'

libiminuit = Extension('iminuit._libiminuit',
                       sources=['iminuit/_libiminuit' + ext] + minuit_src,
                       include_dirs=[minuit_header],
                       libraries=[],
                       # extra_compile_args=['-Wall', '-Wno-sign-compare',
                       #                      '-Wno-write-strings'],
                       extra_link_args=[])
extensions = [libiminuit]

if USE_CYTHON:
    extensions = cythonize(extensions)


# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/en/latest/development.html#single-sourcing-the-version-across-setup-py-and-your-project
# This is one of the recommended methods that works in Python 2 and 3:
def get_version():
    version = {}
    with open("iminuit/info.py") as fp:
        exec(fp.read(), version)
    return version['__version__']


__version__ = get_version()

long_description = ''.join(open('README.rst').readlines()[4:])

setup(
    name='iminuit',
    version=__version__,
    description='MINUIT from Python - Fitting like a boss',
    long_description=long_description,
    author='Piti Ongmongkolkul',
    author_email='piti118@gmail.com',
    url='https://github.com/iminuit/iminuit',
    download_url='http://pypi.python.org/packages/source/i/'
                 'iminuit/iminuit-%s.tar.gz' % __version__,
    package_dir={'iminuit': 'iminuit'},
    packages=['iminuit', 'iminuit.frontends', 'iminuit.tests'],
    ext_modules=extensions,
    install_requires=['setuptools'],
    extras_require={
        'all': ['numpy', 'ipython', 'matplotlib'],
    },
    tests_require=['pytest', 'pytest-cov'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'License :: OSI Approved :: MIT License'
    ],
    cmdclass={
        'test': PyTest,
        # 'coverage': CoverageCommand,
    }
)
