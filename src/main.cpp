#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_application(py::module);
void bind_contours(py::module);
void bind_fcn(py::module);
void bind_functionminimum(py::module);
void bind_hesse(py::module);
void bind_migrad(py::module);
void bind_minuitparameter(py::module);
void bind_print(py::module);
void bind_strategy(py::module);
void bind_userparameterstate(py::module);

PYBIND11_MODULE(_core, m) {
  bind_application(m);
  bind_contours(m);
  bind_fcn(m);
  bind_functionminimum(m);
  bind_migrad(m);
  bind_minuitparameter(m);
  bind_print(m);
  bind_strategy(m);
  bind_userparameterstate(m);
}
