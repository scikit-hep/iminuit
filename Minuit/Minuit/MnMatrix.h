#ifndef MN_MnMatrix_H_
#define MN_MnMatrix_H_

//add MnConfig file to define before everything compiler 
// dependent macros

#include "Minuit/MnConfig.h"

// Removing this the following include will cause the library to fail
// to compile with gcc 4.0.0 under Red Hat Enterprise Linux 3.  That
// is, FumiliBuiilder.cpp will fail with message about ambigous enum.
// Putting an include <vector> before other includes in that file will
// fix it, but then another file class will fail with the same
// message.  I don't understand it, but putting the include <vector>
// in this one spot, fixes the problem and does not require any other
// changes to the source code.
//
// Paul_Kunz@slac.stanford.edu  3 June 2005
//
#include <vector>

#include "Minuit/LASymMatrix.h"
#include "Minuit/LAVector.h"
#include "Minuit/LaInverse.h"
#include "Minuit/LaOuterProduct.h"

typedef LASymMatrix MnAlgebraicSymMatrix;
typedef LAVector MnAlgebraicVector;

#endif //MN_MnMatrix_H_
