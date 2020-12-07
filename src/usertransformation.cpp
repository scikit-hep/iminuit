#include <Minuit2/MnUserTransformation.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>

namespace py = pybind11;
using namespace ROOT::Minuit2;

int size(const MnUserTransformation& self) {
  return static_cast<int>(self.Parameters().size());
}

auto iter(const MnUserTransformation& self) {
  return py::make_iterator(self.Parameters().begin(), self.Parameters().end());
}

const auto& getitem(const MnUserTransformation& self, int i) {
  if (i < 0) i += size(self);
  if (i < 0 || i >= size(self)) throw py::index_error();
  return self.Parameter(i);
}

void bind_usertransformation(py::module m) {
  py::class_<MnUserTransformation>(m, "MnUserTransformation")

      .def(py::init<>())

      .def("name", &MnUserTransformation::GetName)
      .def("index", &MnUserTransformation::FindIndex)

      .def("__len__", size)
      .def("__iter__", iter)
      .def("__getitem__", getitem)

      ;
}
