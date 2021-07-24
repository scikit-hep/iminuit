#include <Minuit2/MnUserTransformation.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>

namespace py = pybind11;
using namespace ROOT::Minuit2;

static_assert(std::is_standard_layout<MnUserTransformation>(), "");

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
      .def("ext2int", &MnUserTransformation::Ext2int)
      .def("int2ext", &MnUserTransformation::Int2ext)
      .def("dint2ext", &MnUserTransformation::DInt2Ext)
      .def("ext_of_int", &MnUserTransformation::ExtOfInt)
      .def("int_of_ext", &MnUserTransformation::IntOfExt)
      .def_property_readonly("variable_parameters",
                             &MnUserTransformation::VariableParameters)

      .def("__len__", size)
      .def("__iter__", iter)
      .def("__getitem__", getitem)

      .def(py::pickle(
          [](const MnUserTransformation& self) {
            const auto d = reinterpret_cast<const Layout*>(&self);
            return py::make_tuple(self.Precision().Eps(), self.Parameters(),
                                  d->fExtOfInt, self.InitialParValues());
          },
          [](py::tuple tp) {
            if (tp.size() != 4) throw std::runtime_error("invalid state");

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
