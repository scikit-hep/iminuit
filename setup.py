from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np

RTMinuit = Extension('RTMinuit._libRTMinuit',
                    sources = ['RTMinuit/_libRTMinuit.cpp'],
                    include_dirs= [np.get_include()],
                    libraries = ['lcg_Minuit'],
                    extra_compile_args = ['-Wno-write-strings'],
                    extra_link_args = [])

setup (
    name = 'RTMinuit',
    version = '1.00',
    description = 'Another Minuit wrapper',
    author='Piti Ongmongkolkul',
    author_email='piti118@gmail.com',
    url='https://github.com/piti118/RTMinuit',
    package_dir = {'RTMinuit': 'RTMinuit'},
    packages = ['RTMinuit'],
    ext_modules = [RTMinuit]
       )