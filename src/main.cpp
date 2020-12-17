#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_application(py::module);
void bind_contours(py::module);
void bind_fcn(py::module);
void bind_functionminimum(py::module);
void bind_hesse(py::module);
void bind_machineprecision(py::module);
void bind_migrad(py::module);
void bind_minimumstate(py::module);
void bind_minos(py::module);
void bind_minuitparameter(py::module);
void bind_print(py::module);
void bind_scan(py::module);
void bind_simplex(py::module);
void bind_strategy(py::module);
void bind_usercovariance(py::module);
void bind_userparameterstate(py::module);
void bind_usertransformation(py::module);

PYBIND11_MODULE(_core, m) {
  bind_application(m);
  bind_contours(m);
  bind_fcn(m);
  bind_functionminimum(m);
  bind_hesse(m);
  bind_machineprecision(m);
  bind_migrad(m);
  bind_minimumstate(m);
  bind_minos(m);
  bind_minuitparameter(m);
  bind_print(m);
  bind_scan(m);
  bind_simplex(m);
  bind_strategy(m);
  bind_usercovariance(m);
  bind_userparameterstate(m);
  bind_usertransformation(m);
}
