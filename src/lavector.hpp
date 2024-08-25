#ifndef IMINUIT_LAVECTOR
#define IMINUIT_LAVECTOR

#include "pybind11.hpp"
#include <Minuit2/LAVector.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

py::list lavector2py(const LAVector& v);
LAVector py2lavector(py::list ls);

#endif
