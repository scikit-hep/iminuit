from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np
from os.path import dirname, join
from glob import glob
#static link
cwd = dirname(__file__)
minuit_src = glob(join(cwd,'Minuit/src/*.cpp'))
minuit_header = join(cwd,'Minuit')
RTMinuit = Extension('RTMinuit._libRTMinuit',
                    sources = ['RTMinuit/_libRTMinuit.cpp'] + minuit_src,
                    include_dirs= [np.get_include(),minuit_header],
                    libraries = [],
                    extra_compile_args = ['-Wno-write-strings'],
                    extra_link_args = [])

execfile('RTMinuit/info.py')

setup (
    name = 'RTMinuit',
    version = __version__,
    description = 'Another Minuit wrapper',
    author='Piti Ongmongkolkul',
    author_email='piti118@gmail.com',
    url='https://github.com/piti118/RTMinuit',
    package_dir = {'RTMinuit': 'RTMinuit'},
    packages = ['RTMinuit'],
    ext_modules = [RTMinuit]
    )