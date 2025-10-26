cmake_minimum_required(VERSION 3.15...3.26)

project(iminuit LANGUAGES CXX)

if(NOT DEFINED SKBUILD)
  message(
    WARNING
      "You should call this through Python so that all variables are set; pip install -v ."
  )
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

set(CMAKE_CXX_STANDARD
    14
    CACHE STRING "C++ version selection") # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON) # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF) # optional, keep compiler extensionsn off
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB SOURCES_A "src/*.cpp")
file(GLOB SOURCES_B "extern/root/math/minuit2/src/*.cxx")

pybind11_add_module(_core MODULE ${SOURCES_A} ${SOURCES_B})

target_include_directories(_core PRIVATE extern/root/core/foundation/inc
                                         extern/root/math/minuit2/inc)
target_compile_definitions(_core PUBLIC PYBIND11_DETAILED_ERROR_MESSAGES=1)
set_target_properties(_core PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
if(ipo_supported)
  set_target_properties(_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Add compiler-specific extra warnings
if(MSVC)
  target_compile_options(_core PRIVATE /std:c++17 /Y-)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # lots of warnings from gcc, including odr violations
  target_compile_options(
    _core
    PRIVATE -Wall
            -Wextra
            -Werror
            -Werror=odr
            -Werror=lto-type-mismatch
            -Werror=strict-aliasing
            -pedantic
            -fstrict-aliasing)
  target_link_options(_core PRIVATE -Werror -Wodr -Wlto-type-mismatch)
else()
  # lots of warnings from clang
  target_compile_options(_core PRIVATE -Wall -Wextra -pedantic -Werror
                                       -Werror=odr -fstrict-aliasing)
  target_link_options(_core PRIVATE -Werror -Wodr)
endif()

install(TARGETS _core DESTINATION iminuit)
