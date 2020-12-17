#include "lavector.hpp"

namespace py = pybind11;
using namespace ROOT::Minuit2;

py::list lavector2py(const LAVector& self) {
  py::list ls;
  for (unsigned i = 0; i < self.size(); ++i) ls.append(self.Data()[i]);
  return ls;
}

LAVector py2lavector(py::list ls) {
  LAVector v(ls.size());
  for (unsigned i = 0; i < v.size(); ++i) v.Data()[i] = ls[i].cast<double>();
  return v;
}
