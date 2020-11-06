# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import os
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            import re

            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute() / "iminuit"
        # required for auto-detection of auxiliary "native" libs
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + f"{extdir}/",
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        # build_args = ["--config", cfg]
        build_args = []

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            njobs = self.parallel if self.parallel else 2
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j{}".format(njobs)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=os.environ
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


cwd = Path(__file__).parent

# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/guides/single-sourcing-package-version/?highlight=single%20sourcing
with (cwd / "src/iminuit/version.py").open() as fp:
    version = {}
    exec(fp.read(), version)  # this loads __version__
    version = version["__version__"]


with (cwd / "README.rst").open() as fp:
    txt = fp.read()
    # skip everything up to the skip marker
    skip_marker = ".. skip-marker-do-not-remove"
    long_description = txt[txt.index(skip_marker) + len(skip_marker) :].lstrip()


setup(
    name="iminuit",
    maintainer="Hans Dembinski",
    maintainer_email="hans.dembinski@gmail.com",
    description="Jupyter-friendly Python frontend for MINUIT2 in C++",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="http://github.com/scikit-hep/iminuit",
    author="Piti Ongmongkolkul and the iminuit team",
    download_url="https://pypi.python.org/pypi/iminuit",
    project_urls={
        "Documentation": "https://iminuit.readthedocs.io",
        "Source Code": "http://github.com/scikit-hep/iminuit",
    },
    license="MIT",
    packages=["iminuit", "iminuit.tests"],
    package_dir={"": "src"},
    version=version,
    python_requires=">=3.6",
    zip_safe=False,
    ext_modules=[CMakeExtension("cmake_example")],
    cmdclass=dict(build_ext=CMakeBuild),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
)
