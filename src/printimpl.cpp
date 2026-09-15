#include "Minuit2/MnPrint.h"
#include "pybind11.hpp"

using ROOT::Minuit2::MnPrint;

void MnPrint::Impl(MnPrint::Verbosity level, const std::string& s) {
  const char* label[5] = {"E", "W", "I", "D", "T"};
  const int ilevel = static_cast<int>(level);
  pybind11::print(label[ilevel], s);
}
