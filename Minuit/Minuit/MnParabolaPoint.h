#ifndef MN_MnParabolaPoint_H_
#define MN_MnParabolaPoint_H_


/** 

A point of a parabola.

<p>

????!!!! in reality it is just a general point in two dimensional space,
there is nothing that would indicate, that it belongs to a parabola.
This class defines simpy an (x,y) pair!!!!

@author Fred James and Matthias Winkler; comments added by Andras Zsenei
and Lorenzo Moneta

@ingroup Minuit

\todo Should it be called MnParabolaPoint or just Point?

 */


class MnParabolaPoint {

public:


  /**
    
  Initializes the point with its coordinates.

  @param x the x (first) coordinate of the point.
  @param y the y (second) coordinate of the point. 

  */

  MnParabolaPoint(double x, double y) : theX(x), theY(y) {}

  ~MnParabolaPoint() {}


  /**

  Accessor to the x (first) coordinate.

  @return the x (first) coordinate of the point.

  */

  double x() const {return theX;}


  /**

  Accessor to the y (second) coordinate.

  @return the y (second) coordinate of the point.

  */

  double y() const {return theY;}

private:

  double theX;
  double theY;
};

#endif //MN_MnParabolaPoint_H_
