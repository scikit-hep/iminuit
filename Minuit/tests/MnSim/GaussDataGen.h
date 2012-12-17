#ifndef MN_GaussDataGen_H_
#define MN_GaussDataGen_H_

#include <vector>

class GaussDataGen {

public:

  GaussDataGen(unsigned int npar = 100);

  ~GaussDataGen() {}

  std::vector<double> positions() const {return thePositions;}
  std::vector<double> measurements() const {return theMeasurements;}
  std::vector<double> variances() const {return theVariances;}

  double sim_mean() const {return theSimMean;}
  double sim_var() const {return theSimVar;}
  double sim_const() const {return 1.;}

private:

  double theSimMean;
  double theSimVar;
  std::vector<double> thePositions;
  std::vector<double> theMeasurements;
  std::vector<double> theVariances;
};

#endif //MN_GaussDataGen_H_
