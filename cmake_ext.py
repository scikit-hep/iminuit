import sys
import os
import platform
import subprocess
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, sourcedir=""):
        Extension.__init__(self, "CMakeExtension", sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute() / "iminuit"
        # required for auto-detection of auxiliary "native" libs
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}/",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
        build_args = ["--config", cfg]  # needed by some generators, e.g. on Windows

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            is_x86 = sys.maxsize > 2 ** 32
            cmake_args += ["-A", "x64" if is_x86 else "Win32"]
            build_args += ["--", "/m:1"]
        else:
            njobs = self.parallel or os.cpu_count() or 1
            build_args += ["--", f"-j{njobs}"]

        print(f"cmake args: {' '.join(cmake_args)}")
        print(f"build args: {' '.join(build_args)}")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
