#include "pybind11.hpp"
#include <Minuit2/MnUserTransformation.h>
#include <type_traits>

namespace py = pybind11;
using namespace ROOT::Minuit2;

static_assert(std::is_standard_layout<MnUserTransformation>(), "");

// We cannot define this struct inside the body of the lambda,
// MSVC will crash if we do
struct Layout {
  MnMachinePrecision fPrecision;

  std::vector<MinuitParameter> fParameters;
  std::vector<unsigned int> fExtOfInt;

  SinParameterTransformation fDoubleLimTrafo;
  SqrtUpParameterTransformation fUpperLimTrafo;
  SqrtLowParameterTransformation fLowerLimTrafo;

  mutable std::vector<double> fCache;
};

int size(const MnUserTransformation& self) {
  return static_cast<int>(self.Parameters().size());
}

auto iter(const MnUserTransformation& self) {
  return py::make_iterator(self.Parameters().begin(), self.Parameters().end());
}

// Minuit2 only asserts on the index, which is compiled out in release builds
unsigned ext_index(const MnUserTransformation& self, int i) {
  const int n = size(self);
  if (i < 0) i += n;
  if (i < 0 || i >= n) throw py::index_error();
  return static_cast<unsigned>(i);
}

unsigned int_index(const MnUserTransformation& self, int i) {
  const int n = static_cast<int>(self.VariableParameters());
  if (i < 0) i += n;
  if (i < 0 || i >= n) throw py::index_error();
  return static_cast<unsigned>(i);
}

const auto& getitem(const MnUserTransformation& self, int i) {
  return self.Parameter(ext_index(self, i));
}

void bind_usertransformation(py::module m) {
  py::class_<MnUserTransformation>(m, "MnUserTransformation")

      .def(py::init<>())

      .def("name",
           [](const MnUserTransformation& self, int i) -> const std::string& {
             return self.GetName(ext_index(self, i));
           })
      .def("index", &MnUserTransformation::FindIndex)
      .def("ext2int", [](const MnUserTransformation& self, int i,
                         double x) { return self.Ext2int(ext_index(self, i), x); })
      .def("int2ext", [](const MnUserTransformation& self, int i,
                         double x) { return self.Int2ext(int_index(self, i), x); })
      .def("dint2ext", [](const MnUserTransformation& self, int i,
                          double x) { return self.DInt2Ext(int_index(self, i), x); })
      .def("ext_of_int", [](const MnUserTransformation& self,
                            int i) { return self.ExtOfInt(int_index(self, i)); })
      .def("int_of_ext", [](const MnUserTransformation& self,
                            int i) { return self.IntOfExt(ext_index(self, i)); })
      .def_property_readonly("variable_parameters",
                             &MnUserTransformation::VariableParameters)

      .def("__len__", size)
      .def("__iter__", iter, py::keep_alive<0, 1>())
      .def("__getitem__", getitem)

      .def(py::pickle(
          [](const MnUserTransformation& self) {
            const auto d = reinterpret_cast<const Layout*>(&self);
            return py::make_tuple(self.Precision().Eps(), self.Parameters(),
                                  d->fExtOfInt, self.InitialParValues());
          },
          [](py::tuple tp) {
            if (tp.size() != 4)
              throw std::runtime_error("MnUserTransformation invalid state");

            MnUserTransformation tr;
            tr.SetPrecision(tp[0].cast<double>());

            // evil workaround, will segfault or cause UB if source layout changes
            auto d = reinterpret_cast<Layout*>(&tr);
            d->fParameters = tp[1].cast<std::vector<MinuitParameter>>();
            d->fExtOfInt = tp[2].cast<std::vector<unsigned>>();
            d->fCache = tp[3].cast<std::vector<double>>();
            return tr;
          }))

      ;
}
