#ifndef MN_MnLineSearch_H_
#define MN_MnLineSearch_H_

#include "Minuit/MnMatrix.h"

class MnFcn;
class MinimumParameters;
class MnMachinePrecision;
class MnParabolaPoint;




/** 

Implements a 1-dimensional minimization along a given direction 
(i.e. quadratic interpolation) It is independent of the algorithm 
that generates the direction vector. It brackets the 1-dimensional 
minimum and iterates to approach the real minimum of the n-dimensional
function.


@author Fred James and Matthias Winkler; comments added by Andras Zsenei
and Lorenzo Moneta

@ingroup Minuit

*/




class MnLineSearch  {

public:

  MnLineSearch() {}

  ~MnLineSearch() {}

  MnParabolaPoint operator()(const MnFcn&, const MinimumParameters&, const MnAlgebraicVector&, double, const MnMachinePrecision&) const;

private:

};

#endif //MN_MnLineSearch_H_
