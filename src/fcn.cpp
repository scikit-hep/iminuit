#include "fcn.hpp"
#include <Minuit2/FCNGradientBase.h>
#include <Minuit2/MnPrint.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <sstream>
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
    return check_value(py::cast<double>(fcn_(a)), x);
  }
  return check_value(py::cast<double>(fcn_(*py::cast(x))), x);
}

std::vector<double> FCN::Gradient(const std::vector<double>& x) const {
  ++ngrad_;
  if (use_array_call_) {
    py::array_t<double> a(static_cast<ssize_t>(x.size()), x.data());
    return check_vector(py::cast<std::vector<double>>(grad_(a)), x);
  }
  return check_vector(py::cast<std::vector<double>>(grad_(*py::cast(x))), x);
}

std::string error_message(const std::vector<double>& x) {
  std::ostringstream msg;
  msg << "result is NaN for [ ";
  for (auto&& xi : x) msg << xi << " ";
  msg << "]";
  return msg.str();
}

double FCN::check_value(double r, const std::vector<double>& x) const {
  if (std::isnan(r)) {
    if (throw_nan_)
      throw std::runtime_error(error_message(x));
    else {
      MnPrint("FCN").Warn([&](std::ostream& os) { os << error_message(x); });
    }
  }
  return r;
}

std::vector<double> FCN::check_vector(std::vector<double> r,
                                      const std::vector<double>& x) const {
  bool has_nan = false;
  for (auto&& ri : r) has_nan |= std::isnan(ri);
  if (has_nan) {
    if (throw_nan_)
      throw std::runtime_error(error_message(x));
    else {
      MnPrint("FCN::Gradient").Warn([&](std::ostream& os) { os << error_message(x); });
    }
  }
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
