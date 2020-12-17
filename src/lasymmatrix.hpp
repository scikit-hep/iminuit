#ifndef IMINUIT_LASYMMATRIX
#define IMINUIT_LASYMMATRIX

#include <Minuit2/LASymMatrix.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

py::tuple lasymmatrix2py(const LASymMatrix& self);

LASymMatrix py2lasymmatrix(py::tuple tp);

#endif
