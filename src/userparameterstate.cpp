#include <Minuit2/MnUserParameterState.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

int size(const MnUserParameterState& self) {
  return static_cast<int>(self.MinuitParameters().size());
}

const MinuitParameter& getitem(const MnUserParameterState& self, int i) {
  const int n = size(self);
  if (i < 0) i += n;
  if (i >= n) throw py::index_error();
  return self.Parameter(i);
}

auto iter(const MnUserParameterState& self) {
  return py::make_iterator(self.MinuitParameters().begin(),
                           self.MinuitParameters().end());
}

py::object globalcc(const MnUserParameterState& self) {
  auto gcc = self.GlobalCC();
  if (gcc.IsValid()) return py::cast(gcc.GlobalCC());
  return py::cast(nullptr);
}

void bind_userparameterstate(py::module m) {
  py::class_<MnUserParameterState>(m, "MnUserParameterState")

      .def(py::init<>())
      .def(py::init<const MnUserParameterState&>())

      // .def_property_readonly("parameters", &MnUserParameterState::Parameters)
      // .def_property_readonly("covariance", &MnUserParameterState::Covariance)
      .def("add",
           py::overload_cast<const std::string&, double>(&MnUserParameterState::Add))
      .def("add", py::overload_cast<const std::string&, double, double>(
                      &MnUserParameterState::Add))
      .def("add", py::overload_cast<const std::string&, double, double, double, double>(
                      &MnUserParameterState::Add))
      .def("fix", py::overload_cast<unsigned>(&MnUserParameterState::Fix))
      .def("release", py::overload_cast<unsigned>(&MnUserParameterState::Release))
      .def("set_value",
           py::overload_cast<unsigned, double>(&MnUserParameterState::SetValue))
      .def("set_error",
           py::overload_cast<unsigned, double>(&MnUserParameterState::SetError))
      .def("set_limits", py::overload_cast<unsigned, double, double>(
                             &MnUserParameterState::SetLimits))
      .def("set_upper_limit",
           py::overload_cast<unsigned, double>(&MnUserParameterState::SetUpperLimit))
      .def("set_lower_limit",
           py::overload_cast<unsigned, double>(&MnUserParameterState::SetLowerLimit))
      .def("remove_limits",
           py::overload_cast<unsigned>(&MnUserParameterState::RemoveLimits))
      .def_property_readonly("fval", &MnUserParameterState::Fval)
      .def_property_readonly("edm", &MnUserParameterState::Edm)
      .def_property_readonly("covariance", &MnUserParameterState::Covariance)
      .def_property_readonly("globalcc", globalcc)
      .def_property_readonly("is_valid", &MnUserParameterState::IsValid)
      .def_property_readonly("has_covariance", &MnUserParameterState::HasCovariance)
      .def_property_readonly("has_globalcc", &MnUserParameterState::HasGlobalCC)
      .def("__len__", size)
      .def("__getitem__", getitem)
      .def("__iter__", iter)

      ;
}

// TODO implemenet equal comparison
// # cdef states_equal(n, MnUserParameterStateConstPtr a, MnUserParameterStateConstPtr
// b): #     result = False #     for i in range(n): #         result |=
// a.Parameter(i).Value() != b.Parameter(i).Value() #         result |=
// a.Parameter(i).Error() != b.Parameter(i).Error() #         result |=
// a.Parameter(i).IsFixed() != b.Parameter(i).IsFixed() #         result |=
// a.Parameter(i).HasLowerLimit() != b.Parameter(i).HasLowerLimit() #         result |=
// a.Parameter(i).HasUpperLimit() != b.Parameter(i).HasUpperLimit() #         result |=
// a.Parameter(i).LowerLimit() != b.Parameter(i).LowerLimit() #         result |=
// a.Parameter(i).UpperLimit() != b.Parameter(i).UpperLimit() #     return result
