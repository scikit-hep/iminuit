# setup compiles with -O2 or -O3 by default, use
# CFLAGS="-O0" python setup.py ... to override for debugging

import os
import sys
import platform
from os.path import dirname, join, exists
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import CCompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import MSVCCompiler
import distutils.ccompiler

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

# turn off warnings raised by Minuit and generated Cython code that need
# to be fixed in the original code bases of Minuit and Cython
compiler_opts = {
    CCompiler: {},
    UnixCCompiler: {
    'extra_compile_args': [
        '-Wno-shorten-64-to-32', '-Wno-parentheses',
        '-Wno-unused-variable', '-Wno-sign-compare',
        '-Wno-cpp',  # suppresses #warnings from numpy
        '-Wno-deprecated-declarations',  # suppresses warnings about auto_ptr
    ]
    + ['--coverage'] if bool(os.environ.get("COVERAGE", False)) else []
    + ['-stdlib=libc++'] if platform.system() == "Darwin" else [],
    'extra_link_args': ['--coverage'] if bool(os.environ.get("COVERAGE", False)) else []
    + ['-stdlib=libc++'] if platform.system() == "Darwin" else []
    },
    MSVCCompiler: {'extra_compile_args': [
        '/EHsc',
    ]},
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


# prevent setup from recompiling static Minuit2 code again and again
def lazy_compile(self, sources, output_dir=None, macros=None,
                 include_dirs=None, debug=0, extra_preargs=None,
                 extra_postargs=None, depends=None):
    macros, objects, extra_postargs, pp_opts, build = \
        self._setup_compile(output_dir, macros, include_dirs, sources,
                            depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    for obj in objects:
        try:
            src, ext = build[obj]
        except KeyError:
            continue
        if not exists(obj) or os.stat(obj).st_mtime < os.stat(src).st_mtime:
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    return objects


# monkey-patching lazy_compile into CCompiler
distutils.ccompiler.CCompiler.compile = lazy_compile

# Static linking
cwd = dirname(__file__)
minuit_src = glob(join(cwd, 'Minuit/src/*.cxx'))
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
        raise SystemExit('Looks like you are installing iminuit from github. '
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
                       define_macros=[('WARNINGMSG', '1')])
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

with open('README.rst') as readme_rst:
    txt = readme_rst.read()
    # skip everything up to the skip marker
    skip_marker = ".. skip-marker-do-not-remove"
    long_description = txt[txt.index(skip_marker) + len(skip_marker):].lstrip()

setup(
    name='iminuit',
    version=__version__,
    description='MINUIT2 from Python - Fitting like a boss',
    long_description=long_description,
    author='Piti Ongmongkolkul and others',
    maintainer='Hans Dembinski',
    maintainer_email='hans.dembinski@gmail.com',
    url='https://github.com/iminuit/iminuit',
    download_url='http://pypi.python.org/packages/source/i/'
                 'iminuit/iminuit-%s.tar.gz' % __version__,
    packages=['iminuit', 'iminuit.tests'],
    ext_modules=extensions,
    install_requires=['numpy'],
    setup_requires = [] + pytest_runner,
    extras_require={
        'tests': ['pytest', 'cython', 'matplotlib', 'scipy'],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
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
        'build_ext': SmartBuildExt,
    }
)
