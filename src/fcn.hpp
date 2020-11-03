#include <Minuit2/FCNGradientBase.h>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;

/**
   Function interface for Minuit2. Calls the underlying Python function which
   computes the objective function. The interface of this class is defined by the
   abstract base class FCNGradientBase, which itself derives from FCNBase.

   It calls the function with the parameter vector converted into positional arguments
   or a single Numpy array, depending on the value of use_array_call_.
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
