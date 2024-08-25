#ifndef IMINUIT_LASYMMATRIX
#define IMINUIT_LASYMMATRIX

#include "pybind11.hpp"
#include <Minuit2/LASymMatrix.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

py::tuple lasymmatrix2py(const LASymMatrix& self);

LASymMatrix py2lasymmatrix(py::tuple tp);

#endif
