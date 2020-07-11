# Use CFLAGS="-g -Og -DDEBUG" python setup.py ... for debugging

import os
import platform
from os.path import dirname, join, exists
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import CCompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import MSVCCompiler
import distutils.ccompiler

extra_flags = []
if bool(os.environ.get("COVERAGE", False)):
    extra_flags += ["--coverage"]
if platform.system() == "Darwin":
    extra_flags += ["-stdlib=libc++"]

# turn off warnings raised by Minuit and generated Cython code that need
# to be fixed in the original code bases of Minuit and Cython
compiler_opts = {
    CCompiler: {},
    UnixCCompiler: {
        "extra_compile_args": [
            "-std=c++11",
            "-Wno-shorten-64-to-32",
            "-Wno-parentheses",
            "-Wno-unused-variable",
            "-Wno-sign-compare",
            "-Wno-cpp",  # suppresses #warnings from numpy
            "-Wno-deprecated-declarations",
        ]
        + extra_flags,
        "extra_link_args": extra_flags,
    },
    MSVCCompiler: {"extra_compile_args": ["/EHsc"]},
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
def lazy_compile(
    self,
    sources,
    output_dir=None,
    macros=None,
    include_dirs=None,
    debug=0,
    extra_preargs=None,
    extra_postargs=None,
    depends=None,
):
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )

    pp_opts += compiler_opts.get(self, {}).get("extra_compile_args", [])
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

# We follow the recommendation how to distribute Cython modules:
# http://docs.cython.org/src/reference/compilation.html#distributing-cython-modules
try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    if exists("src/iminuit/_libiminuit.cpp"):
        print("Cython is not available ... using pre-generated cpp file.")
    else:
        raise SystemExit(
            "Looks like you are compiling iminuit from sources. "
            "This requires Cython. Run\n\n"
            "   pip install cython\n\n"
            "for a system-wide installation, or\n\n"
            "   pip install --user cython\n\n"
            "for a user-wide installation."
        )

ext = ".pyx" if USE_CYTHON else ".cpp"

try:
    import numpy

    numpy_header = [numpy.get_include()]
except ImportError:
    numpy_header = []


# Install missing Minuit2 submodule as needed
if not os.listdir(join(cwd, "extern/Minuit2")):
    try:
        import subprocess as subp

        print("Minuit2 submodule is missing, attempting download...")
        subp.check_call(["git", "submodule", "update"])
    except subp.CalledProcessError:
        raise SystemExit(
            "Could not download Minuit2 submodule, run `git submodule update` manually"
        )

minuit2_cxx = [
    join(cwd, "extern/Minuit2/src", x) + ".cxx"
    for x in open(join(cwd, "minuit2_cxx.lst"), "r").read().split("\n")
    if x
]

libiminuit = Extension(
    "iminuit._libiminuit",
    sources=sorted(glob(join(cwd, "src/iminuit/*" + ext)) + minuit2_cxx),
    include_dirs=[join(cwd, "extern/Minuit2/inc")] + numpy_header,
    define_macros=[
        ("WARNINGMSG", "1"),
        ("ROOT_Math_VecTypes", "1"),
        ("MATH_NO_PLUGIN_MANAGER", "1"),
    ],
)
extensions = [libiminuit]

if USE_CYTHON:
    extensions = cythonize(extensions)


# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/guides/single-sourcing-package-version/?highlight=single%20sourcing
with open(join(cwd, "src/iminuit/version.py")) as fp:
    version = {}
    exec(fp.read(), version)  # this loads __version__
    version = version["__version__"]

with open(join(cwd, "README.rst")) as readme_rst:
    txt = readme_rst.read()
    # skip everything up to the skip marker
    skip_marker = ".. skip-marker-do-not-remove"
    long_description = txt[txt.index(skip_marker) + len(skip_marker) :].lstrip()

setup(
    name="iminuit",
    version=version,
    description="Jupyter-friendly Python frontend for MINUIT2 in C++",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Piti Ongmongkolkul and the iminuit team",
    maintainer="Hans Dembinski",
    maintainer_email="hans.dembinski@gmail.com",
    url="http://github.com/scikit-hep/iminuit",
    project_urls={
        "Documentation": "https://iminuit.readthedocs.io",
        "Source Code": "http://github.com/scikit-hep/iminuit",
    },
    packages=["iminuit", "iminuit.tests"],
    package_dir={"": "src"},
    ext_modules=extensions,
    install_requires=["numpy>=1.11.3"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "License :: OSI Approved :: MIT License",
    ],
    cmdclass={"build_ext": SmartBuildExt},
)
