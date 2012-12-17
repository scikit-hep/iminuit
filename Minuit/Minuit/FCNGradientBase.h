#ifndef MN_FCNGradientBase_H_
#define MN_FCNGradientBase_H_

#include "Minuit/FCNBase.h"

/** Extension of the FCNBase for providing the analytical gradient of the 
    function. The user-gradient is checked at the beginning of the 
    minimization against the Minuit internal numerical gradient in order to 
    spot problems in the analytical gradient calculation. This can be turned 
    off by overriding checkGradient() to make it return "false".
    The size of the output gradient vector must be equal to the size of the 
    input parameter vector.
    Minuit does a check of the user gradient at the beginning, if this is not 
    wanted the method "checkGradient()" has to be overridden to return 
    "false".
 */

class FCNGradientBase : public FCNBase {

public:

  virtual ~FCNGradientBase() {}

  virtual std::vector<double> gradient(const std::vector<double>&) const = 0;

  virtual bool checkGradient() const {return true;}

};

#endif //MN_FCNGradientBase_H_
