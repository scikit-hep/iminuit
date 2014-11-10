# -*- coding: utf8 -*-
from distutils.core import setup, Extension
from os.path import dirname, join
from glob import glob

# Static linking
cwd = dirname(__file__)
minuit_src = glob(join(cwd, 'Minuit/src/*.cpp'))
minuit_header = join(cwd, 'Minuit')

libiminuit = Extension('iminuit._libiminuit',
                       sources=['iminuit/_libiminuit.cpp'] + minuit_src,
                       include_dirs=[minuit_header],
                       libraries=[],
                       #extra_compile_args=['-Wall', '-Wno-sign-compare',
                       #                      '-Wno-write-strings'],
                       extra_link_args=[])


# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/en/latest/development.html#single-sourcing-the-version-across-setup-py-and-your-project
# This is one of the recommended methods that works in Python 2 and 3:
def get_version():
    version = {}
    with open("iminuit/info.py") as fp:
        exec(fp.read(), version)
    return version['__version__']

__version__ = get_version()


setup(
    name='iminuit',
    version=__version__,
    description='Interactive Minimization Tools based on MINUIT',
    long_description=''.join(open('README.rst').readlines()[4:]),
    author='Piti Ongmongkolkul',
    author_email='piti118@gmail.com',
    url='https://github.com/iminuit/iminuit',
    download_url='http://pypi.python.org/packages/source/i/'
                 'iminuit/iminuit-%s.tar.gz' % __version__,
    package_dir={'iminuit': 'iminuit'},
    packages=['iminuit', 'iminuit.frontends'],
    ext_modules=[libiminuit],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.3',
        #'Programming Language :: Python :: 3.4',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'License :: OSI Approved :: MIT License'
    ]
)
