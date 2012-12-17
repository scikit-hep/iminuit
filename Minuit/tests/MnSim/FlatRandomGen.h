#ifndef MN_FlatRandomGen_H_
#define MN_FlatRandomGen_H_

#include <math.h>

class FlatRandomGen {

public:

  FlatRandomGen() : theMean(0.5), theDelta(0.5) {}

  FlatRandomGen(double mean, double delta) : theMean(mean), theDelta(delta) {}

  ~FlatRandomGen() {}

  double mean() const {return theMean;}

  double delta() const {return theDelta;}
  
  double operator()() const {
    return 2.*delta()*(std::rand()/double(RAND_MAX) - 0.5) + mean();
  }

private:

  double theMean;
  double theDelta;
};

#endif //MN_FlatRandomGen_H_
