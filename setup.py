# -*- coding: utf8 -*-
from distutils.core import setup, Extension
from os.path import dirname, join
from glob import glob

#static link
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

execfile('iminuit/info.py')

setup(
    name='iminuit',
    version=__version__,
    description='Interactive Minimization Tools based on MINUIT',
    long_description=''.join(open('README.rst').readlines()[4:]),
    author='Piti Ongmongkolkul',
    author_email='piti118@gmail.com',
    url='http://iminuit.github.io/iminuit/',
    download_url='http://pypi.python.org/packages/source/i/'
            'iminuit/iminuit-%s.tar.gz' % __version__,
    package_dir={'iminuit': 'iminuit'},
    packages=['iminuit'],
    ext_modules=[libiminuit],
    classifiers=[
        "Programming Language :: Python",
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'License :: OSI Approved :: MIT License'
    ]
)
