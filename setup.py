from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np

FCN = Extension('RTMinuit.FCN',
                    sources = ['RTMinuit/FCN.c'],
                    include_dirs= [np.get_include()],
                    extra_link_args = [])

setup (name = 'RTMinuit',
       version = '1.00',
       description = 'Root TMinuit wrapper',
       author='Piti Ongmongkolkul',
       author_email='piti118@gmail.com',
       url='https://github.com/piti118/RTMinuit',
       package_dir = {'RTMinuit': 'RTMinuit'},
       packages = ['RTMinuit'],
       ext_modules = [FCN]
       )