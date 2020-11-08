import sys
import os
import platform
import subprocess
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import multiprocessing


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
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

        cfg = "Debug" if self.debug else "Release"
        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            is_x86 = sys.maxsize > 2 ** 32
            cmake_args += ["-A", "x64" if is_x86 else "Win32"]
            build_args += ["--", "/m"]
        else:
            njobs = self.parallel if self.parallel else multiprocessing.cpu_count()
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
