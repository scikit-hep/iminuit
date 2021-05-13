from pathlib import Path
from setuptools import setup
import sys

cwd = Path(__file__).parent

sys.path.append(str(cwd))
from cmake_ext import CMakeExtension, CMakeBuild  # noqa: E402

setup(
    zip_safe=False,
    ext_modules=[CMakeExtension("iminuit._core")],
    cmdclass={"build_ext": CMakeBuild},
)
