#include "fcn.hpp"
#include "pybind11.hpp"
#include <Minuit2/FCNGradientBase.h>
#include <Minuit2/MnPrint.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

namespace py = pybind11;
using namespace ROOT::Minuit2;

FCN::FCN(py::object fcn, py::object grad, py::object g2, py::object hessian,
         bool array_call, double errordef)
    : fcn_{fcn}
    , grad_{grad}
    , g2_{g2}
    , hessian_{hessian}
    , array_call_{array_call}
    , errordef_{errordef} {
  auto util = py::module_::import("iminuit.util");
  auto address_of_cfunc = util.attr("_address_of_cfunc");
  auto address = py::cast<std::uintptr_t>(address_of_cfunc(fcn_));
  if (address) {
    MnPrint("FCN").Debug("using cfunc");
    cfcn_ = reinterpret_cast<cfcn_t>(address);
    array_call_ = true;
  }
}

double FCN::operator()(const std::vector<double>& x) const {
  ++nfcn_;
  if (array_call_) {
    if (cfcn_) {
      return cfcn_(x.size(), x.data());
    } else {
      py::array_t<double> a(static_cast<py::ssize_t>(x.size()), x.data());
      return check_value(py::cast<double>(fcn_(a)), x);
    }
  }
  return check_value(py::cast<double>(fcn_(*py::cast(x))), x);
}

std::vector<double> FCN::Gradient(const std::vector<double>& x) const {
  ++ngrad_;
  if (array_call_) {
    py::array_t<double> a(static_cast<py::ssize_t>(x.size()), x.data());
    return check_vector(py::cast<std::vector<double>>(grad_(a)), x);
  }
  return check_vector(py::cast<std::vector<double>>(grad_(*py::cast(x))), x);
}

std::vector<double> FCN::G2(const std::vector<double>& x) const {
  ++ng2_;
  if (array_call_) {
    py::array_t<double> a(static_cast<py::ssize_t>(x.size()), x.data());
    return check_vector(py::cast<std::vector<double>>(g2_(a)), x);
  }
  return check_vector(py::cast<std::vector<double>>(g2_(*py::cast(x))), x);
}

std::vector<double> FCN::Hessian(const std::vector<double>& x) const {
  ++nhessian_;
  if (array_call_) {
    py::array_t<double> a(static_cast<py::ssize_t>(x.size()), x.data());
    // TODO convert properly from a 2d numpy array on python side
    return check_vector(py::cast<std::vector<double>>(hessian_(a)), x);
  }
  // TODO convert properly from a 2d numpy array on python side
  return check_vector(py::cast<std::vector<double>>(hessian_(*py::cast(x))), x);
}

double FCN::Up() const { return errordef_; }

void set_errordef(FCN& self, double value) {
  if (value > 0) {
    self.errordef_ = value;
  } else
    throw std::invalid_argument("errordef must be a positive number");
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

double FCN::ndata() const {
  if (py::hasattr(fcn_, "ndata")) return py::cast<double>(fcn_.attr("ndata"));
  return std::numeric_limits<double>::quiet_NaN();
}

void bind_fcn(py::module m) {

  py::class_<FCNBase>(m, "FCNBase");
  py::class_<FCN, FCNBase>(m, "FCN")

      .def(py::init<py::object, py::object, py::object, py::object, bool, double>())

      .def("gradient", &FCN::Gradient)
      .def("_ndata", &FCN::ndata)
      .def_readwrite("_nfcn", &FCN::nfcn_)
      .def_readwrite("_ngrad", &FCN::ngrad_)
      .def_readwrite("_ng2", &FCN::ng2_)
      .def_readwrite("_nhessian", &FCN::nhessian_)
      .def_readwrite("_throw_nan", &FCN::throw_nan_)
      .def_property("_errordef", &FCN::Up, &set_errordef)
      .def_readonly("_array_call", &FCN::array_call_)
      .def_property_readonly("_cfcn", [](const FCN& self) { return bool(self.cfcn_); })
      .def_readonly("_fcn", &FCN::fcn_)
      .def_readonly("_grad", &FCN::grad_)
      .def_readonly("_g2", &FCN::g2_)
      .def_readonly("_hessian", &FCN::hessian_)

      .def("__call__", &FCN::operator())

      .def(py::pickle(
          [](const FCN& self) {
            return py::make_tuple(self.fcn_, self.grad_, self.g2_, self.hessian_,
                                  self.array_call_, self.errordef_, self.throw_nan_,
                                  self.nfcn_, self.ngrad_, self.ng2_, self.nhessian_);
          },
          [](py::tuple tp) {
            if (tp.size() != 11) throw std::runtime_error("FCN invalid state");
            FCN fcn{
                tp[0], tp[1], tp[2], tp[3], tp[4].cast<bool>(), tp[5].cast<double>()};
            fcn.throw_nan_ = tp[6].cast<bool>();
            fcn.nfcn_ = tp[7].cast<unsigned>();
            fcn.ngrad_ = tp[8].cast<unsigned>();
            fcn.ng2_ = tp[9].cast<unsigned>();
            fcn.nhessian_ = tp[10].cast<unsigned>();
            return fcn;
          }))

      ;
}
