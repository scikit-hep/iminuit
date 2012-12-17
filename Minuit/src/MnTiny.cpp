#include "Minuit/MnTiny.h"

double MnTiny::one() const {return theOne;}

double MnTiny::operator()(double epsp1) const {
  double result = epsp1 - one();
  return result;
}
