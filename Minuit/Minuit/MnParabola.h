#ifndef MN_MnParabola_H_
#define MN_MnParabola_H_

#include <math.h>

/** 

This class defines a parabola of the form a*x*x + b*x + c

@author Fred James and Matthias Winkler; comments added by Andras Zsenei
and Lorenzo Moneta

@ingroup Minuit

 */

class MnParabola {

public:


  /**

  Constructor that initializes the parabola with its three parameters.

  @param a the coefficient of the quadratic term
  @param b the coefficient of the linear term
  @param c the constant

  */

  MnParabola(double a, double b, double c) : theA(a), theB(b), theC(c) {}


  ~MnParabola() {}


  /**

  Evaluates the parabola a the point x.

  @param x the coordinate where the parabola needs to be evaluated.

  @return the y coordinate of the parabola corresponding to x.

  */

  double y(double x) const {return (theA*x*x + theB*x +theC);}


  /**

  Calculates the bigger of the two x values corresponding to the 
  given y value.

  <p>

  ???????!!!!!!!!! And when there is none?? it looks like it will
  crash?? what is sqrt (-1.0) ?

  @param y the y value for which the x value is to be calculated.
  
  @return the bigger one of the two corresponding values.

  */

  // ok, at first glance it does not look like the formula for the quadratic 
  // equation, but it is!  ;-)
  double x_pos(double y) const {return (sqrt(y/theA + min()*min() - theC/theA) + min());}
  // maybe it is worth to check the performance improvement with the below formula??
  //   double x_pos(double y) const {return (sqrt(y/theA + theB*theB/(4.*theA*theA) - theC/theA)  - theB/(2.*theA));}



  /**

  Calculates the smaller of the two x values corresponding to the 
  given y value.

  <p>

  ???????!!!!!!!!! And when there is none?? it looks like it will
  crash?? what is sqrt (-1.0) ?

  @param y the y value for which the x value is to be calculated.
  
  @return the smaller one of the two corresponding values.

  */

  double x_neg(double y) const {return (-sqrt(y/theA + min()*min() - theC/theA) + min());}


  /**

  Calculates the x coordinate of the minimum of the parabola.

  @return x coordinate of the minimum.

  */

  double min() const {return -theB/(2.*theA);}


  /**

  Calculates the y coordinate of the minimum of the parabola.

  @return y coordinate of the minimum.

  */

  double ymin() const {return (-theB*theB/(4.*theA) + theC);}


  /**

  Accessor to the coefficient of the quadratic term. 
  
  @return the coefficient of the quadratic term.

   */

  double a() const {return theA;}


  /**

  Accessor to the coefficient of the linear term. 

  @return the coefficient of the linear term. 

  */

  double b() const {return theB;}


  /**

  Accessor to the coefficient of the constant term. 

  @return the coefficient of the constant term.

  */

  double c() const {return theC;}

private:

  double theA;
  double theB;
  double theC;
};

#endif //MN_MnParabola_H_
