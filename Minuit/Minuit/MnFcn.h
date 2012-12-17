#ifndef MN_MnFcn_H_
#define MN_MnFcn_H_

#include "Minuit/MnConfig.h"
#include "Minuit/MnMatrix.h"

#include <vector>

class FCNBase;

class MnFcn {

public:

  MnFcn(const FCNBase& fcn) : theFCN(fcn), theNumCall(0) {}

  virtual ~MnFcn();

  virtual double operator()(const MnAlgebraicVector&) const;
  unsigned int numOfCalls() const {return theNumCall;}

  //
  //forward interface
  //
  double errorDef() const;
  double up() const;

  const FCNBase& fcn() const {return theFCN;}

private:

  const FCNBase& theFCN;

protected:

  mutable int theNumCall;
};

#endif //MN_MnFcn_H_
