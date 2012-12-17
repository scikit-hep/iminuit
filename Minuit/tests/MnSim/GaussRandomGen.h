#ifndef MN_GaussRandomGen_H_
#define MN_GaussRandomGen_H_

#include <math.h>

class GaussRandomGen {

public:

  GaussRandomGen() : theMean(0.), theSigma(1.) {}

  GaussRandomGen(double mean, double sigma) : theMean(mean), theSigma(sigma) {}

  ~GaussRandomGen() {}

  double mean() const {return theMean;}

  double sigma() const {return theSigma;}

  double operator()() const {
    //need to random variables flat in [0,1)
    double r1 = std::rand()/double(RAND_MAX);
    double r2 = std::rand()/double(RAND_MAX);

    //two possibilities to generate a random gauss variable (m=0,s=1)
    double s = sqrt(-2.*log(1.-r1))*cos(2.*M_PI*r2);
//     double s = sqrt(-2.*log(1.-r1))*sin(2.*M_PI*r2);

    //scale to desired gauss
    return sigma()*s + mean();
  }

private:

  double theMean;
  double theSigma;
  
};

#endif //MN_GaussRandomGen_H_
