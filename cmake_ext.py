import sys
import os
import subprocess
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        # required for auto-detection of auxiliary "native" libs
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}/",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
        build_args = ["--config", cfg]  # needed by some generators, e.g. on Windows

        if self.compiler.compiler_type == "msvc":
            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in ("ARM", "Win64"))

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not contains_arch:
                # Convert distutils Windows platform specifiers to CMake -A arguments
                arch = {
                    "win32": "Win32",
                    "win-amd64": "x64",
                    "win-arm32": "ARM",
                    "win-arm64": "ARM64",
                }[self.plat_name]
                cmake_args += ["-A", arch]

            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # CMake 3.12+ only.
            njobs = self.parallel or os.cpu_count() or 1
            build_args += [f"-j{njobs}"]

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
