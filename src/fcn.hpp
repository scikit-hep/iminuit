#include "pybind11.hpp"
#include <Minuit2/FCNGradientBase.h>
#include <cstdint>
#include <vector>

namespace py = pybind11;

/**
   Function interface for Minuit2. Calls the underlying Python function which
   computes the objective function. The interface of this class is defined by
   FCNBase.

   It calls the function with the parameter vector converted into positional
   arguments or by passing a single Numpy array, depending on the value of
   array_call_.
*/
struct FCN : ROOT::Minuit2::FCNBase {
  FCN(py::object fcn, py::object grad, py::object g2, py::object hessian,
      bool array_call, double errordef);

  double operator()(const std::vector<double>& x) const override;
  std::vector<double> Gradient(const std::vector<double>&) const override;
  std::vector<double> G2(std::vector<double> const&) const override;
  std::vector<double> Hessian(std::vector<double> const&) const override;

  double Up() const override;

  bool HasGradient() const override { return !grad_.is_none(); }
  bool HasG2() const override { return !g2_.is_none(); }
  bool HasHessian() const override { return !hessian_.is_none(); }

  double check_value(double r, const std::vector<double>& x) const;
  std::vector<double> check_vector(std::vector<double> r, const std::vector<double>& x,
                                   const char* label, unsigned size) const;

  double ndata() const;

  py::object fcn_, grad_, g2_, hessian_;
  bool array_call_;
  mutable double errordef_;
  using cfcn_t = double (*)(std::uint32_t, const double*);
  cfcn_t cfcn_ = nullptr;
  bool throw_nan_ = false;
  mutable unsigned nfcn_ = 0, ngrad_ = 0, ng2_ = 0, nhessian_ = 0;
};
