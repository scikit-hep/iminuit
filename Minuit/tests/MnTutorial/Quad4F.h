#include "Minuit/FCNBase.h"

class Quad4F : public FCNBase {

public:

  Quad4F() {}

  ~Quad4F() {}

  double operator()(const std::vector<double>& par) const {

    double x = par[0];
    double y = par[1];
    double z = par[2];
    double w = par[3];

    return ( (1./70.)*(21*x*x + 20*y*y + 19*z*z - 14*x*z - 20*y*z) + w*w );
  }

  double up() const {return 1.;}

private:

};
