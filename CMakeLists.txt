cmake_minimum_required(VERSION 3.18)

project(iminuit CXX)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
# include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensionsn off

# Currently, Scikit-build does not support FindPython, so we convert the
# provided hints ourselves.
if(SKBUILD)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARY "${PYTHON_LIBRARY}")
endif()

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Scikit-Build does not add your site-packages to the search path automatically,
# so we need to add it here.
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import skbuild, os; print(os.path.dirname(os.path.dirname(skbuild.__file__)))"
  OUTPUT_VARIABLE _tmp_sitepackages
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT ECHO_OUTPUT_VARIABLE
  ECHO_ERROR_VARIABLE)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_sitepackages}")

# Now we can find pybind11
add_subdirectory(extern/pybind11)
add_subdirectory(extern/root/math/minuit2)

pybind11_add_module(_core MODULE src/main.cpp)
target_link_libraries(_core PRIVATE Minuit2::Math)

install(TARGETS _core DESTINATION .)
