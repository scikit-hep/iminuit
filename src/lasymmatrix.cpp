#include "lasymmatrix.hpp"

namespace py = pybind11;
using namespace ROOT::Minuit2;

py::tuple lasymmatrix2py(const LASymMatrix& self) {
  py::list ls;
  for (unsigned i = 0; i < self.size(); ++i) ls.append(self.Data()[i]);
  return py::make_tuple(self.Nrow(), ls);
}

LASymMatrix py2lasymmatrix(py::tuple tp) {
  LASymMatrix v(tp[0].cast<unsigned>());
  auto ls = tp[1].cast<py::list>();
  for (unsigned i = 0; i < v.size(); ++i) v.Data()[i] = ls[i].cast<double>();
  return v;
}
