#ifndef MN_FumiliFCNBase_H_
#define MN_FumiliFCNBase_H_

#include "Minuit/FCNBase.h"
#include "Minuit/ParametricFunction.h"

/** 
 
Extension of the FCNBase for the Fumili method. Fumili applies only to 
minimization problems used for fitting. The method is based on a 
linearization of the model function negleting second derivatives. 
User needs to provide the model function. The figure-of-merit describing
the difference between the model function and the actual measurements
has to be implemented by the user in a subclass of FumiliFCNBase.
For an example see the FumiliChi2FCN and FumiliStandardChi2FCN classes.


@author  Andras Zsenei and Lorenzo Moneta, Creation date: 23 Aug 2004

@see <A HREF="http://www.cern.ch/winkler/minuit/tutorial/mntutorial.pdf">MINUIT Tutorial</A> on function minimization, section 5

@see FumiliChi2FCN

@see FumiliStandardChi2FCN

@ingroup Minuit

 */



class FumiliFCNBase : public FCNBase {

public:

  /**
     Default Constructor. Need in this case to create when implementing evaluateAll the gradient and hessian vectors with the right size
  */

  FumiliFCNBase()  : theNumberOfParameters(0) {}

  /**

  Constructor which initializes the class with the function provided by the
  user for modeling the data.

  @param npar the number of parameters 

  */


  FumiliFCNBase(unsigned int npar) 
  {
    initAndReset(npar);
  }



//   FumiliFCNBase(const ParametricFunction& modelFCN) { theModelFunction = &modelFCN; }



  virtual ~FumiliFCNBase() {}




  /**
  
  Evaluate function value, gradient and hessian using Fumili approximation, for values of parameters p
  The resul is cached inside and is return from the FumiliFCNBase::value ,  FumiliFCNBase::gradient and 
  FumiliFCNBase::hessian methods 

  @param par vector of parameters

  **/

  virtual  void evaluateAll( const std::vector<double> & par ) = 0; 


  /**
   Return cached value of objective function estimated previously using the  FumiliFCNBase::evaluateAll method

  **/

  virtual double value() const { return theValue; } 

  /**
   Return cached value of function gradient estimated previously using the  FumiliFCNBase::evaluateAll method
  **/

  virtual const std::vector<double> & gradient() const { return theGradient; }

  /**
   Return value of the i-th j-th element of the hessian matrix estimated previously using the  FumiliFCNBase::evaluateAll method
   @param row row index of the matrix
   @param col col index of the matrix
  **/

  virtual double hessian(unsigned int row, unsigned int col) const { 
    assert( row < theGradient.size() && col < theGradient.size() ); 
    if(row > col) 
      return theHessian[col+row*(row+1)/2];
    else
      return theHessian[row+col*(col+1)/2];    
  }

  /**
     return number of function variable (parameters) , i.e. function dimension
   */

  virtual unsigned int dimension() { return theNumberOfParameters; }

protected : 

  /**
     initialize and reset values of gradien and hessian
   */

  virtual void initAndReset(unsigned int npar) {
    theNumberOfParameters = npar;
    theGradient = std::vector<double>(npar); 
    theHessian = std::vector<double>(static_cast<int>( 0.5*npar*(npar+1) ));      
  }

  // methods to be used by the derived classes to set the values 
  void setFCNValue(double value) { theValue = value; }

  std::vector<double> & gradient() { return theGradient; }

  std::vector<double> & hessian() { return theHessian; }



 private:

  // A pointer to the model function which describes the data
  const ParametricFunction *theModelFunction;

  // define these data members protected because can be modified by the derived classes 

 private: 

  double theValue; 
  std::vector<double> theGradient; 
  unsigned int theNumberOfParameters; 
  std::vector<double> theHessian;


};

#endif //MN_FumiliFCNBase_H_
