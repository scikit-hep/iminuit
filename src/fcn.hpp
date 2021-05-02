#include <Minuit2/FCNGradientBase.h>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;

/**
   Function interface for Minuit2. Calls the underlying Python function which
   computes the objective function. The interface of this class is defined by the
   abstract base class FCNGradientBase, which itself derives from FCNBase.

   It calls the function with the parameter vector converted into positional arguments
   or by passing a single Numpy array, depending on the value of array_call_.
*/
struct FCN : ROOT::Minuit2::FCNGradientBase {
  FCN(py::object fcn, py::object grad, bool array_call, double errordef);

  double operator()(const std::vector<double>& x) const override;
  std::vector<double> Gradient(const std::vector<double>&) const override;
  bool CheckGradient() const override { return false; }

  double Up() const override;
  void SetUp(double x) { errordef_ = x; }

  double check_value(double r, const std::vector<double>& x) const;
  std::vector<double> check_vector(std::vector<double> r,
                                   const std::vector<double>& x) const;

  double ndata() const;

  py::object fcn_, grad_;
  bool array_call_;
  mutable double errordef_;
  using cfcn_t = double (*)(std::uint32_t, const double*);
  cfcn_t cfcn_ = nullptr;
  bool throw_nan_ = false;
  mutable unsigned nfcn_ = 0, ngrad_ = 0;
};
