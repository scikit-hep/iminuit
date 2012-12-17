#ifndef MN_GaussFcn2_H_
#define MN_GaussFcn2_H_

#include "Minuit/FCNBase.h"

#include <vector>

class GaussFcn2 : public FCNBase {

public:

  GaussFcn2(const std::vector<double>& meas,
	  const std::vector<double>& pos,
	  const std::vector<double>& mvar) : theMeasurements(meas),
					     thePositions(pos),
					     theMVariances(mvar), 
					     theMin(0.) {init();}
  ~GaussFcn2() {}

  virtual void init();

  virtual double up() const {return 1.;}
  virtual double operator()(const std::vector<double>&) const;
  virtual double errorDef() const {return up();}
  
  std::vector<double> measurements() const {return theMeasurements;}
  std::vector<double> positions() const {return thePositions;}
  std::vector<double> variances() const {return theMVariances;}
  
private:

  std::vector<double> theMeasurements;
  std::vector<double> thePositions;
  std::vector<double> theMVariances;
  double theMin;
};

#endif //MN_GaussFcn2_H_
