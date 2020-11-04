#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MinosError.h>
#include <Minuit2/MnMinos.h>
#include <Minuit2/MnStrategy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_minos(py::module m) {
  py::class_<MinosError>(m, "MinosError")

      .def_property_readonly("number", &MinosError::Parameter)
      .def_property_readonly("lower", &MinosError::Lower)
      .def_property_readonly("upper", &MinosError::Upper)
      .def_property_readonly("is_valid", &MinosError::IsValid)
      .def_property_readonly("lower_valid", &MinosError::LowerValid)
      .def_property_readonly("upper_valid", &MinosError::UpperValid)
      .def_property_readonly("at_lower_limit", &MinosError::AtLowerLimit)
      .def_property_readonly("at_upper_limit", &MinosError::AtUpperLimit)
      .def_property_readonly("at_lower_max_fcn", &MinosError::AtLowerMaxFcn)
      .def_property_readonly("at_upper_max_fcn", &MinosError::AtUpperMaxFcn)
      .def_property_readonly("lower_new_min", &MinosError::LowerNewMin)
      .def_property_readonly("upper_new_min", &MinosError::UpperNewMin)
      .def_property_readonly("nfcn", &MinosError::NFcn)
      .def_property_readonly("min", &MinosError::Min)

      ;

  py::class_<MnMinos>(m, "MnMinos")

      .def(py::init<const FCNBase&, const FunctionMinimum&, const MnStrategy&>())
      // int ipar, unsigned maxcalls, double toler
      .def("__call__", &MnMinos::Minos)

      ;
}
