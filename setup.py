# setup compiles with -O2 or -O3 by default, use
# CFLAGS="-O0" python setup.py ... to override

import sys
from os.path import dirname, join, exists
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    # turn off warnings raised by Minuit and generated Cython code that need
    # to be fixed in the original code bases of Minuit and Cython
    c_opts = {
        'msvc': [
            '/EHsc',
            ],
        'unix': [
            '-Wno-shorten-64-to-32', '-Wno-parentheses',
            '-Wno-unused-variable', '-Wno-sign-compare',
            '-Wno-cpp',  # suppresses #warnings from numpy
            '-std=c++11', # Minuit2 requires C++11 (as ROOT does)
        ]
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


# Static linking
cwd = dirname(__file__)
minuit_src = glob(join(cwd, 'Minuit/src/*.cxx'))
minuit_src += glob(join(cwd, 'Minuit/src/math/*.cxx'))
minuit_header = [join(cwd, 'Minuit/inc')]

# We follow the recommendation how to distribute Cython modules:
# http://docs.cython.org/src/reference/compilation.html#distributing-cython-modules
try:
    from Cython.Build import cythonize
    USE_CYTHON = True  # TODO: add command line option?
except ImportError:
    USE_CYTHON = False
    if exists('iminuit/_libiminuit.cpp'):
        print('Cython is not available ... using pre-generated cpp file.')
    else:
        raise SystemExit('Looks like you are installing iminuit from github.'
                         'This requires Cython. Run\n\n'
                         '   pip install cython\n\n'
                         'for a system-wide installation, or\n\n'
                         '   pip install --user cython\n\n'
                         'for a user-wide installation.')

ext = '.pyx' if USE_CYTHON else '.cpp'

try:
    import numpy

    numpy_header = [numpy.get_include()]
except ImportError:
    numpy_header = []

libiminuit = Extension('iminuit._libiminuit',
                       sources=(glob(join(cwd, 'iminuit/*' + ext)) + minuit_src),
                       include_dirs=minuit_header + numpy_header,
                       language='c++',
                       define_macros=[('WARNINGMSG', None),
                                      ('MATH_NO_PLUGIN_MANAGER', None)])
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
    packages=['iminuit', 'iminuit.frontends', 'iminuit.tests'],
    ext_modules=extensions,
    install_requires=['setuptools', 'numpy'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'License :: OSI Approved :: MIT License',
    ],
    cmdclass={
        'build_ext': BuildExt,
    }
)
