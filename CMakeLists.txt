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

file(GLOB SOURCES_A "src/*.cpp")
set(SOURCES_B
  extern/root/math/minuit2/src/AnalyticalGradientCalculator.cxx
  extern/root/math/minuit2/src/BasicMinimumError.cxx
  extern/root/math/minuit2/src/BFGSErrorUpdator.cxx
  extern/root/math/minuit2/src/DavidonErrorUpdator.cxx
  extern/root/math/minuit2/src/HessianGradientCalculator.cxx
  extern/root/math/minuit2/src/InitialGradientCalculator.cxx
  extern/root/math/minuit2/src/LaEigenValues.cxx
  extern/root/math/minuit2/src/LaInnerProduct.cxx
  extern/root/math/minuit2/src/LaInverse.cxx
  extern/root/math/minuit2/src/LaOuterProduct.cxx
  extern/root/math/minuit2/src/LaSumOfElements.cxx
  extern/root/math/minuit2/src/LaVtMVSimilarity.cxx
  extern/root/math/minuit2/src/MPIProcess.cxx
  extern/root/math/minuit2/src/MinimumBuilder.cxx
  extern/root/math/minuit2/src/MnApplication.cxx
  extern/root/math/minuit2/src/MnContours.cxx
  extern/root/math/minuit2/src/MnCovarianceSqueeze.cxx
  extern/root/math/minuit2/src/MnFcn.cxx
  extern/root/math/minuit2/src/MnFunctionCross.cxx
  extern/root/math/minuit2/src/MnGlobalCorrelationCoeff.cxx
  extern/root/math/minuit2/src/MnHesse.cxx
  extern/root/math/minuit2/src/MnLineSearch.cxx
  extern/root/math/minuit2/src/MnMachinePrecision.cxx
  extern/root/math/minuit2/src/MnMinos.cxx
  extern/root/math/minuit2/src/MnParabolaFactory.cxx
  extern/root/math/minuit2/src/MnPlot.cxx
  extern/root/math/minuit2/src/MnPosDef.cxx
  extern/root/math/minuit2/src/MnPrint.cxx
  extern/root/math/minuit2/src/MnPrintImpl.cxx
  extern/root/math/minuit2/src/MnSeedGenerator.cxx
  extern/root/math/minuit2/src/MnStrategy.cxx
  extern/root/math/minuit2/src/MnTiny.cxx
  extern/root/math/minuit2/src/MnTraceObject.cxx
  extern/root/math/minuit2/src/MnUserFcn.cxx
  extern/root/math/minuit2/src/MnUserParameterState.cxx
  extern/root/math/minuit2/src/MnUserParameters.cxx
  extern/root/math/minuit2/src/MnUserTransformation.cxx
  extern/root/math/minuit2/src/ModularFunctionMinimizer.cxx
  extern/root/math/minuit2/src/NegativeG2LineSearch.cxx
  extern/root/math/minuit2/src/Numerical2PGradientCalculator.cxx
  extern/root/math/minuit2/src/SinParameterTransformation.cxx
  extern/root/math/minuit2/src/SqrtLowParameterTransformation.cxx
  extern/root/math/minuit2/src/SqrtUpParameterTransformation.cxx
  extern/root/math/minuit2/src/VariableMetricBuilder.cxx
  extern/root/math/minuit2/src/VariableMetricEDMEstimator.cxx
  extern/root/math/minuit2/src/mnbins.cxx
  extern/root/math/minuit2/src/mndasum.cxx
  extern/root/math/minuit2/src/mndaxpy.cxx
  extern/root/math/minuit2/src/mnddot.cxx
  extern/root/math/minuit2/src/mndscal.cxx
  extern/root/math/minuit2/src/mndspmv.cxx
  extern/root/math/minuit2/src/mndspr.cxx
  extern/root/math/minuit2/src/mnlsame.cxx
  extern/root/math/minuit2/src/mnteigen.cxx
  extern/root/math/minuit2/src/mntplot.cxx
  extern/root/math/minuit2/src/mnvert.cxx
  extern/root/math/minuit2/src/mnxerbla.cxx
)

set(SOURCES )

pybind11_add_module(_core MODULE
  ${SOURCES_A}
  ${SOURCES_B}
)
target_include_directories(_core PRIVATE extern/root/math/minuit2/inc extern/root/math/mathcore/inc)

install(TARGETS _core DESTINATION .)
