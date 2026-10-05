cmake_minimum_required(VERSION 3.15...4.0)

project(iminuit LANGUAGES CXX)

if(NOT DEFINED SKBUILD)
  message(
    WARNING
      "You should call this through Python so that all variables are set; pip install -v ."
  )
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
# Set before pybind11_add_module() so it skips linking its own pybind11::lto
# target
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ${ipo_supported})

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ version selection") # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON) # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF) # optional, keep compiler extensions off
find_package(pybind11 CONFIG REQUIRED)

file(GLOB SOURCES_A CONFIGURE_DEPENDS "src/*.cpp")
# Keep in sync with tool.scikit-build.sdist.include in pyproject.toml
set(SOURCES_B
    extern/Minuit2/src/math/Util.cxx
    extern/Minuit2/src/AnalyticalGradientCalculator.cxx
    extern/Minuit2/src/BFGSErrorUpdator.cxx
    extern/Minuit2/src/DavidonErrorUpdator.cxx
    extern/Minuit2/src/HessianGradientCalculator.cxx
    extern/Minuit2/src/ExternalInternalGradientCalculator.cxx
    extern/Minuit2/src/InitialGradientCalculator.cxx
    extern/Minuit2/src/MPIProcess.cxx
    extern/Minuit2/src/MinimumBuilder.cxx
    extern/Minuit2/src/MnApplication.cxx
    extern/Minuit2/src/MnContours.cxx
    extern/Minuit2/src/MnCovarianceSqueeze.cxx
    extern/Minuit2/src/MnFcn.cxx
    extern/Minuit2/src/MnFunctionCross.cxx
    extern/Minuit2/src/MnGlobalCorrelationCoeff.cxx
    extern/Minuit2/src/MnHesse.cxx
    extern/Minuit2/src/MnLineSearch.cxx
    extern/Minuit2/src/MnMachinePrecision.cxx
    extern/Minuit2/src/MnMatrix.cxx
    extern/Minuit2/src/MnMinos.cxx
    extern/Minuit2/src/MnParabolaFactory.cxx
    extern/Minuit2/src/MnParameterScan.cxx
    extern/Minuit2/src/MnParameterTransformation.cxx
    extern/Minuit2/src/MnPlot.cxx
    extern/Minuit2/src/MnPosDef.cxx
    extern/Minuit2/src/MnPrint.cxx
    extern/Minuit2/src/MnSeedGenerator.cxx
    extern/Minuit2/src/MnStrategy.cxx
    extern/Minuit2/src/MnScan.cxx
    extern/Minuit2/src/MnTraceObject.cxx
    extern/Minuit2/src/MnUserParameterState.cxx
    extern/Minuit2/src/MnUserParameters.cxx
    extern/Minuit2/src/MnUserTransformation.cxx
    extern/Minuit2/src/ModularFunctionMinimizer.cxx
    extern/Minuit2/src/NegativeG2LineSearch.cxx
    extern/Minuit2/src/Numerical2PGradientCalculator.cxx
    extern/Minuit2/src/SimplexSeedGenerator.cxx
    extern/Minuit2/src/SimplexBuilder.cxx
    extern/Minuit2/src/SimplexParameters.cxx
    extern/Minuit2/src/ScanBuilder.cxx
    extern/Minuit2/src/VariableMetricBuilder.cxx
    extern/Minuit2/src/VariableMetricEDMEstimator.cxx
    extern/Minuit2/src/mnbins.cxx
    extern/Minuit2/src/mntplot.cxx)
pybind11_add_module(_core MODULE ${SOURCES_A} ${SOURCES_B})

# Keep in sync with tool.scikit-build.sdist.include in pyproject.toml
target_include_directories(_core PRIVATE src extern/Minuit2/inc)

target_compile_definitions(_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1
                                         ROOT_Math_VecTypes MATHCORE_STANDALONE)
set_target_properties(_core PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

# Add compiler-specific extra warnings
if(MSVC)
  target_compile_options(_core PRIVATE /Y-)
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
