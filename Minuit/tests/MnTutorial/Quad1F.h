#include "Minuit/FCNGradientBase.h"

class Quad1F : public FCNGradientBase {

public:

  Quad1F() : theErrorDef(1.) {}

  ~Quad1F() {}

  double operator()(const std::vector<double>& par) const {

    double x = par[0];

    return ( x*x );
  }
  
  std::vector<double> gradient(const std::vector<double>& par) const {
    
    double x = par[0];
    
    return ( std::vector<double>(1, 2.*x) );  
  }

  void setErrorDef(double up) {theErrorDef = up;}

  double up() const {return theErrorDef;}

  const FCNBase* base() const {return this;}

private:
  double theErrorDef;
};

