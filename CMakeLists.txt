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
set(SOURCES_B
    extern/root/math/minuit2/src/AnalyticalGradientCalculator.cxx
    extern/root/math/minuit2/src/BFGSErrorUpdator.cxx
    extern/root/math/minuit2/src/DavidonErrorUpdator.cxx
    extern/root/math/minuit2/src/HessianGradientCalculator.cxx
    extern/root/math/minuit2/src/ExternalInternalGradientCalculator.cxx
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
    extern/root/math/minuit2/src/MnParameterScan.cxx
    extern/root/math/minuit2/src/MnPlot.cxx
    extern/root/math/minuit2/src/MnPosDef.cxx
    extern/root/math/minuit2/src/MnPrint.cxx
    extern/root/math/minuit2/src/MnSeedGenerator.cxx
    extern/root/math/minuit2/src/MnStrategy.cxx
    extern/root/math/minuit2/src/MnScan.cxx
    extern/root/math/minuit2/src/MnTiny.cxx
    extern/root/math/minuit2/src/MnTraceObject.cxx
    extern/root/math/minuit2/src/MnUserFcn.cxx
    extern/root/math/minuit2/src/MnUserParameterState.cxx
    extern/root/math/minuit2/src/MnUserParameters.cxx
    extern/root/math/minuit2/src/MnUserTransformation.cxx
    extern/root/math/minuit2/src/ModularFunctionMinimizer.cxx
    extern/root/math/minuit2/src/NegativeG2LineSearch.cxx
    extern/root/math/minuit2/src/Numerical2PGradientCalculator.cxx
    extern/root/math/minuit2/src/SimplexSeedGenerator.cxx
    extern/root/math/minuit2/src/SimplexBuilder.cxx
    extern/root/math/minuit2/src/SimplexParameters.cxx
    extern/root/math/minuit2/src/SinParameterTransformation.cxx
    extern/root/math/minuit2/src/ScanBuilder.cxx
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
    extern/root/math/minuit2/src/mnxerbla.cxx)

pybind11_add_module(_core MODULE ${SOURCES_A} ${SOURCES_B})

target_include_directories(_core PRIVATE extern/root/math/minuit2/inc)
target_compile_definitions(_core PUBLIC PYBIND11_DETAILED_ERROR_MESSAGES=1)
set_target_properties(_core PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
if(ipo_supported)
  set_target_properties(_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Add compiler-specific extra warnings
if(MSVC)
  target_compile_options(_core PRIVATE /std:c++14 /Y-)
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
