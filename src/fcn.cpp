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

namespace {
// Build a tuple of plain Python floats from the parameter vector. Used to unpack the
// parameters into positional arguments in the scalar (non-array) calling convention.
// This avoids a round-trip through a numpy array (one array allocation plus one
// np.float64 temporary per element) and means user functions receive Python floats.
py::tuple args_as_floats(const std::vector<double>& x) {
  const std::size_t n = x.size();
  py::tuple t(n);
  for (std::size_t i = 0; i < n; ++i) t[i] = py::float_(x[i]);
  return t;
}
} // namespace

std::vector<double> flatten_hessian(py::array_t<double> arg) {
  if (arg.ndim() != 2) throw std::runtime_error("number of dimensions must be 2");

  // flatten the matrix
  auto r = arg.unchecked<2>();
  const unsigned size = arg.shape(0);
  if (arg.shape(1) != size) throw std::runtime_error("2D matrix is not square");
  std::vector<double> result(size * size);
  for (unsigned i = 0; i < size; ++i)
    for (unsigned j = 0; j < size; ++j) result[i * size + j] = r(i, j);
  return result;
}

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

// Convert the value returned by the Python cost function into a double.
//
// In addition to plain Python floats/ints, this accepts numpy scalars and
// numpy arrays of size 1 (0-d or shape (1,)). Older numpy versions allowed
// such arrays to be converted implicitly via float(), but numpy >= 2 raises
// for arrays with ndim > 0, which silently broke user cost functions that
// return e.g. ``np.diff(...)``. We restore the previous behavior by calling
// ``.item()`` on numpy arrays, which works for any size-1 array and raises a
// clear error otherwise.
double as_double(py::handle obj) {
  if (py::isinstance<py::array>(obj)) {
    auto arr = py::reinterpret_borrow<py::array>(obj);
    if (arr.size() != 1)
      throw std::runtime_error(
          "cost function must return a scalar, but returned an array of size " +
          std::to_string(arr.size()));
    return py::cast<double>(arr.attr("item")());
  }
  return py::cast<double>(obj);
}

double FCN::operator()(const std::vector<double>& x) const {
  ++nfcn_;
  if (array_call_) {
    if (cfcn_) {
      return cfcn_(x.size(), x.data());
    } else {
      py::array_t<double> a(static_cast<py::ssize_t>(x.size()), x.data());
      return check_value(as_double(fcn_(a)), x);
    }
  }
  return check_value(as_double(fcn_(*args_as_floats(x))), x);
}

std::vector<double> FCN::Gradient(const std::vector<double>& x) const {
  ++ngrad_;
  const unsigned npar = x.size();
  if (array_call_) {
    py::array_t<double> a(static_cast<py::ssize_t>(npar), x.data());
    return check_vector(py::cast<std::vector<double>>(grad_(a)), x, "Gradient", npar);
  }
  return check_vector(py::cast<std::vector<double>>(grad_(*args_as_floats(x))), x,
                      "Gradient", npar);
}

std::vector<double> FCN::G2(const std::vector<double>& x) const {
  ++ng2_;
  const unsigned npar = x.size();
  if (array_call_) {
    py::array_t<double> a(static_cast<py::ssize_t>(npar), x.data());
    return check_vector(py::cast<std::vector<double>>(g2_(a)), x, "G2", npar);
  }
  return check_vector(py::cast<std::vector<double>>(g2_(*args_as_floats(x))), x, "G2",
                      npar);
}

std::vector<double> FCN::Hessian(const std::vector<double>& x) const {
  ++nhessian_;
  const unsigned npar = x.size();
  if (array_call_) {
    py::array_t<double> a(static_cast<py::ssize_t>(npar), x.data());
    return check_vector(flatten_hessian(hessian_(a)), x, "Hessian", npar * npar);
  }
  // TODO convert properly from a 2d numpy array on python side
  return check_vector(flatten_hessian(hessian_(*args_as_floats(x))), x, "Hessian",
                      npar * npar);
}

double FCN::Up() const { return errordef_; }

void set_errordef(FCN& self, double value) {
  if (value > 0) {
    self.errordef_ = value;
  } else
    throw std::invalid_argument("errordef must be a positive number");
}

std::string result_contains_nan_message(const std::vector<double>& x) {
  std::ostringstream msg;
  msg << "result is NaN for [ ";
  for (auto&& xi : x) msg << xi << " ";
  msg << "]";
  return msg.str();
}

std::string result_has_wrong_size_message(const std::vector<double>& x, unsigned size) {
  std::ostringstream msg;
  msg << "result has size " << x.size() << " but expected size is " << size;
  return msg.str();
}

double FCN::check_value(double r, const std::vector<double>& x) const {
  if (std::isnan(r)) {
    if (throw_nan_)
      throw std::runtime_error(result_contains_nan_message(x));
    else {
      MnPrint("FCN").Warn(
          [&](std::ostream& os) { os << result_contains_nan_message(x); });
    }
  }
  return r;
}

std::vector<double> FCN::check_vector(std::vector<double> r,
                                      const std::vector<double>& x, const char* label,
                                      unsigned size) const {
  if (r.size() != size)
    throw std::runtime_error(result_has_wrong_size_message(r, size));
  bool has_nan = false;
  for (auto&& ri : r) has_nan |= std::isnan(ri);
  if (has_nan) {
    if (throw_nan_)
      throw std::runtime_error(result_contains_nan_message(x));
    else {
      std::string msg("FCN::");
      msg += label;
      MnPrint(msg.c_str()).Warn([&](std::ostream& os) {
        os << result_contains_nan_message(x);
      });
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
