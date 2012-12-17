#include "GaussDataGen.h"
#include "GaussFunction.h"
#include "GaussRandomGen.h"
#include "FlatRandomGen.h"

GaussDataGen::GaussDataGen(unsigned int n) {

  // create random generator for mean values of Gaussian [-50, 50)
  FlatRandomGen rand_mean(0., 50.);
  
  // create random generator for sigma values of Gaussian [1., 11.)
  FlatRandomGen rand_var(6., 5.);
  
  // errors of measurements (Gaussian, mean=0., sig = 0.01)
  double mvariance = 0.01*0.01;
  GaussRandomGen rand_mvar(0., 0.01);
  
  // simulate data
  theSimMean = rand_mean();
  theSimVar = rand_var();
  double sim_sig = sqrt(theSimVar);
  double sim_const = 1.;
  GaussFunction gauss_sim(theSimMean, sim_sig, sim_const);

  for(unsigned int i = 0; i < n; i++) {

    //x-position, from -5sigma < mean < +5sigma
    double position = theSimMean-5.*sim_sig + double(i)*10.*sim_sig/double(n);
    thePositions.push_back(position);

    //y-position (function value)
    double epsilon = rand_mvar();
    theMeasurements.push_back(gauss_sim(position) + epsilon);
    theVariances.push_back(mvariance);
  }

}
