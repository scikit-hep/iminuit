#include <Minuit2/FCNGradientBase.h>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;

/**
    Called by Minuit2 to call the underlying Python function which computes
    the objective function. The interface of this class is defined by the
    abstract base class FCNBase.

    This version calls the function with a tuple of numbers or a single Numpy
    array, depending on the ConvertFunction argument.
*/
struct FCN : ROOT::Minuit2::FCNGradientBase {
  FCN(py::object fcn, py::object grad, bool use_array_call, double up);
  double operator()(const std::vector<double>& x) const override;
  std::vector<double> Gradient(const std::vector<double>&) const override;
  bool CheckGradient() const override { return false; }
  double Up() const override { return up_; }
  void SetUp(double x) { up_ = x; }

  py::object fcn_, grad_;
  bool use_array_call_;
  double up_;
  mutable unsigned nfcn_, ngrad_;
};
