#include "Minuit/MnFcn.h"
#include "Minuit/FCNBase.h"
#include "Minuit/MnVectorTransform.h"

MnFcn::~MnFcn() {
//   std::cout<<"Total number of calls to FCN: "<<theNumCall<<std::endl;
}

double MnFcn::operator()(const MnAlgebraicVector& v) const {

  theNumCall++;
  return theFCN(MnVectorTransform()(v));
}

// double MnFcn::operator()(const std::vector<double>& par) const {
//     return theFCN(par);
// }

double MnFcn::errorDef() const {return theFCN.up();}

double MnFcn::up() const {return theFCN.up();}
