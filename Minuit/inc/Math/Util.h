// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 14 15:44:38 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Utility functions for all ROOT Math classes

#ifndef ROOT_Math_Util
#define ROOT_Math_Util

#include <string>
#include <sstream>

#include <cmath>
#include <limits>
// #include "Types.h"


// for defining unused variables in the interfaces
//  and have still them in the documentation
#define MATH_UNUSED(var)   (void)var


namespace ROOT {

   namespace Math {

   /**
      namespace defining Utility functions needed by mathcore
   */
   namespace Util {

   /**
      Utility function for conversion to strings
   */
   template <class T>
   std::string ToString(const T &val)
   {
      std::ostringstream buf;
      buf << val;

      std::string ret = buf.str();
      return ret;
   }

   /// safe evaluation of log(x) with a protections against negative or zero argument to the log
   /// smooth linear extrapolation below function values smaller than  epsilon
   /// (better than a simple cut-off)

   template<class T>
   inline T EvalLog(T x) {
      static const T epsilon = T(2.0 * std::numeric_limits<double>::min());
#if !defined(R__HAS_VECCORE)
      T logval = x <= epsilon ? x / epsilon + std::log(epsilon) - T(1.0) : std::log(x);
#else
      T logval = vecCore::Blend<T>(x <= epsilon, x / epsilon + std::log(epsilon) - T(1.0), std::log(x));
#endif
      return logval;
   }

   } // end namespace Util

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Util */
