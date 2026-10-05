#include "equal.hpp"
#include "pybind11.hpp"
#include <Minuit2/MnStrategy.h>

namespace ROOT {
namespace Minuit2 {
bool operator==(const MnStrategy& a, const MnStrategy& b) {
  return a.GradientNCycles() == b.GradientNCycles() &&
         a.GradientStepTolerance() == b.GradientStepTolerance() &&
         a.GradientTolerance() == b.GradientTolerance() &&
         a.HessianNCycles() == b.HessianNCycles() &&
         a.HessianStepTolerance() == b.HessianStepTolerance() &&
         a.HessianG2Tolerance() == b.HessianG2Tolerance() &&
         a.HessianGradientNCycles() == b.HessianGradientNCycles() &&
         a.StorageLevel() == b.StorageLevel();
}
} // namespace Minuit2
} // namespace ROOT

namespace py = pybind11;
using namespace ROOT::Minuit2;

// MnStrategy no longer exposes its level, so we remember it here.
struct PyStrategy : MnStrategy {
  unsigned level;
  PyStrategy() : MnStrategy(), level(1) {}
  explicit PyStrategy(unsigned s) : MnStrategy(s), level(s) {}
};

bool operator==(const PyStrategy& a, const PyStrategy& b) {
  return a.level == b.level &&
         static_cast<const MnStrategy&>(a) == static_cast<const MnStrategy&>(b);
}

void set_strategy(PyStrategy& self, unsigned s) {
  if (s > 2) throw std::invalid_argument("invalid strategy");
  self = PyStrategy(s);
}

void bind_strategy(py::module m) {
  // Minuit2 classes take const MnStrategy&, so implicit int conversion must
  // also be registered for the base.
  py::class_<MnStrategy>(m, "_MnStrategyBase").def(py::init<unsigned>());

  py::class_<PyStrategy, MnStrategy>(m, "MnStrategy")

      .def(py::init<>())
      .def(py::init<unsigned>())
      .def_property(
          "strategy", [](const PyStrategy& s) { return s.level; }, set_strategy)

      .def_property("gradient_ncycles", &MnStrategy::GradientNCycles,
                    &MnStrategy::SetGradientNCycles)
      .def_property("gradient_step_tolerance", &MnStrategy::GradientStepTolerance,
                    &MnStrategy::SetGradientStepTolerance)
      .def_property("gradient_tolerance", &MnStrategy::GradientTolerance,
                    &MnStrategy::SetGradientTolerance)
      .def_property("hessian_ncycles", &MnStrategy::HessianNCycles,
                    &MnStrategy::SetHessianNCycles)
      .def_property("hessian_step_tolerance", &MnStrategy::HessianStepTolerance,
                    &MnStrategy::SetHessianStepTolerance)
      .def_property("hessian_g2_tolerance", &MnStrategy::HessianG2Tolerance,
                    &MnStrategy::SetHessianG2Tolerance)
      .def_property("hessian_gradient_ncycles", &MnStrategy::HessianGradientNCycles,
                    &MnStrategy::SetHessianGradientNCycles)
      .def_property("storage_level", &MnStrategy::StorageLevel,
                    &MnStrategy::SetStorageLevel)

      .def(py::self == py::self)

      .def(py::pickle(
          [](const PyStrategy& self) {
            return py::make_tuple(
                self.level, self.GradientNCycles(), self.GradientStepTolerance(),
                self.GradientTolerance(), self.HessianNCycles(),
                self.HessianStepTolerance(), self.HessianG2Tolerance(),
                self.HessianGradientNCycles(), self.StorageLevel());
          },
          [](py::tuple tp) {
            PyStrategy str(tp[0].cast<unsigned>());
            str.SetGradientNCycles(tp[1].cast<unsigned>());
            str.SetGradientStepTolerance(tp[2].cast<double>());
            str.SetGradientTolerance(tp[3].cast<double>());
            str.SetHessianNCycles(tp[4].cast<unsigned>());
            str.SetHessianStepTolerance(tp[5].cast<double>());
            str.SetHessianG2Tolerance(tp[6].cast<double>());
            str.SetHessianGradientNCycles(tp[7].cast<unsigned>());
            str.SetStorageLevel(tp[8].cast<unsigned>());
            return str;
          }))

      ;

  py::implicitly_convertible<unsigned, PyStrategy>();
  py::implicitly_convertible<unsigned, MnStrategy>();
}
