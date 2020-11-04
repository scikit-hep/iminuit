#include "fcn.hpp"
#include <Minuit2/FCNGradientBase.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

namespace py = pybind11;
using namespace ROOT::Minuit2;

FCN::FCN(py::object fcn, py::object grad, bool use_array_call, double up,
         bool throw_nan)
    : fcn_{fcn}
    , grad_{grad}
    , use_array_call_{use_array_call}
    , up_(up)
    , nfcn_{0}
    , ngrad_{0}
    , throw_nan_{throw_nan} {}

double FCN::operator()(const std::vector<double>& x) const {
  ++nfcn_;
  if (use_array_call_) {
    py::array_t<double> a(static_cast<ssize_t>(x.size()), x.data());
    return check_value(py::cast<double>(fcn_(a)));
  }
  return check_value(py::cast<double>(fcn_(*py::cast(x))));
}

std::vector<double> FCN::Gradient(const std::vector<double>& x) const {
  ++ngrad_;
  if (use_array_call_) {
    py::array_t<double> a(static_cast<ssize_t>(x.size()), x.data());
    return check_vector(py::cast<std::vector<double>>(grad_(a)));
  }
  return check_vector(py::cast<std::vector<double>>(grad_(*py::cast(x))));
}

double FCN::check_value(double r) const {
  if (throw_nan_ && std::isnan(r)) throw std::runtime_error("result is NaN");
  return r;
}

std::vector<double> FCN::check_vector(std::vector<double> r) const {
  if (throw_nan_)
    for (auto&& x : r)
      if (std::isnan(x)) throw std::runtime_error("result is NaN");
  return r;
}

void bind_fcn(py::module m) {
  py::class_<FCNBase>(m, "FCNBase");
  py::class_<FCN, FCNBase>(m, "FCN")

      .def(py::init<py::object, py::object, bool, double, bool>())
      .def("__call__", &FCN::operator())
      .def("grad", &FCN::Gradient)
      .def_property("up", &FCN::Up, &FCN::SetUp)
      .def_readonly("use_array_call", &FCN::use_array_call_)
      .def_readwrite("throw_nan", &FCN::throw_nan_)
      .def_readwrite("nfcn", &FCN::nfcn_)
      .def_readwrite("ngrad", &FCN::ngrad_)

      ;
}
