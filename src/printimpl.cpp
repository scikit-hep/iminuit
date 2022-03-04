#include "Minuit2/MnPrint.h"
#include <pybind11/pybind11.h>

using ROOT::Minuit2::MnPrint;

void MnPrint::Impl(MnPrint::Verbosity level, const std::string& s) {
  const char* label[4] = {"E", "W", "I", "D"};
  const int ilevel = static_cast<int>(level);
  pybind11::print(label[ilevel], s);
}
