from pathlib import Path
from setuptools import setup
import sys

cwd = Path(__file__).parent

sys.path.append(str(cwd))
from cmake_ext import CMakeExtension, CMakeBuild  # noqa: E402

with (cwd / "README.rst").open() as fp:
    txt = fp.read()
    # skip everything up to the skip marker
    skip_marker = ".. skip-marker-do-not-remove"
    long_description = txt[txt.index(skip_marker) + len(skip_marker) :].lstrip()

setup(
    long_description=long_description,
    long_description_content_type="text/x-rst",
    zip_safe=False,
    ext_modules=[CMakeExtension("iminuit._core")],
    cmdclass=dict(build_ext=CMakeBuild),
)
