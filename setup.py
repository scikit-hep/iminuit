from pathlib import Path
from setuptools import setup
import sys

cwd = Path(__file__).parent

sys.path.append(str(cwd))
from cmake_ext import CMakeExtension, CMakeBuild  # noqa: E402


# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/guides/single-sourcing-package-version/?highlight=single%20sourcing
with (cwd / "src/iminuit/version.py").open() as fp:
    version = {}
    exec(fp.read(), version)
    version = version["iminuit_version"]


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
    packages=["iminuit"],
    package_dir={"": "src"},
    install_requires=["numpy"],
    version=version,
    python_requires=">=3.6",
    zip_safe=False,
    ext_modules=[CMakeExtension("iminuit._core")],
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
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
