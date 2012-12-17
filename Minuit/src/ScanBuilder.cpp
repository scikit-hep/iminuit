#include "Minuit/ScanBuilder.h"
#include "Minuit/MnParameterScan.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MinimumSeed.h"
#include "Minuit/MinimumState.h"
#include "Minuit/MnFcn.h"

FunctionMinimum ScanBuilder::minimum(const MnFcn& mfcn, const GradientCalculator&, const MinimumSeed& seed, const MnStrategy&, unsigned int, double) const {
  
  MnAlgebraicVector x = seed.parameters().vec();
  MnUserParameterState upst(seed.state(), mfcn.up(), seed.trafo());
  MnParameterScan scan(mfcn.fcn(), upst.parameters(), seed.fval());
  double amin = scan.fval();
  unsigned int n = seed.trafo().variableParameters();
  MnAlgebraicVector dirin(n);
  for(unsigned int i = 0; i < n; i++) {
    unsigned int ext = seed.trafo().extOfInt(i);
    scan(ext);
    if(scan.fval() < amin) {
      amin = scan.fval();
      x(i) = seed.trafo().ext2int(ext, scan.parameters().value(ext));
    }
    dirin(i) = sqrt(2.*mfcn.up()*seed.error().invHessian()(i,i));
  }

  MinimumParameters mp(x, dirin, amin);
  MinimumState st(mp, 0., mfcn.numOfCalls());

  return FunctionMinimum(seed, std::vector<MinimumState>(1, st), mfcn.up());
}
