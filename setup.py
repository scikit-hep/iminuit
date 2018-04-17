# -*- coding: utf-8 -*-
import sys
from os.path import dirname, join
from glob import glob

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand

from distutils.ccompiler import CCompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import MSVCCompiler


# turn off warnings raised by Minuit and generated Cython code that need
# to be fixed in the original code bases of Minuit and Cython
compiler_opts = {
    CCompiler: dict(),
    UnixCCompiler: dict(extra_compile_args=[
            '-Wno-shorten-64-to-32', '-Wno-null-conversion',
            '-Wno-parentheses', '-Wno-unused-variable', '-Wno-sign-compare',
            '-W#warnings'
        ]),
    MSVCCompiler: dict(extra_compile_args=[
            '/EHsc',
        ]),
}


class SmartBuildExt(build_ext):
    def build_extensions(self):
        c = self.compiler
        opts = [v for k, v in compiler_opts.items() if isinstance(c, k)]
        for e in self.extensions:
            for o in opts:
                for attrib, value in o.items():
                    getattr(e, attrib).extend(value)

        build_ext.build_extensions(self)


# http://pytest.org/latest/goodpractices.html#manual-integration
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        # self.pytest_args = '--pyargs iminuit'
        # self.pytest_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.pytest_args = ''

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        import shlex
        errno = pytest.main(shlex.split(self.pytest_args))
        del sys.exitfunc # needed to avoid a bug caused by IPython's exitfunc
        sys.exit(errno)


# Static linking
cwd = dirname(__file__)
minuit_src = glob(join(cwd, 'Minuit/src/*.cxx'))
minuit_header = [join(cwd, 'Minuit/inc')]

try:
    import numpy as np
    minuit_header.append(np.get_include())
    HAVE_NUMPY = 1
except ImportError:
    print('Numpy not available ... Numpy support is disabled')
    HAVE_NUMPY = 0

with open('iminuit/config.pxi', "w") as f:
    f.write("DEF HAVE_NUMPY=%i\n" % HAVE_NUMPY)

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
                       include_dirs=minuit_header,
                       define_macros=[('HAVE_NUMPY', str(HAVE_NUMPY))]
                       )
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
    tests_require=['pytest', 'pytest-cov', 'numpy'],
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
        'build_ext': SmartBuildExt,
    }
)
