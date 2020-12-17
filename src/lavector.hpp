#ifndef IMINUIT_LAVECTOR
#define IMINUIT_LAVECTOR

#include <Minuit2/LAVector.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

py::list lavector2py(const LAVector& v);
LAVector py2lavector(py::list ls);

#endif
