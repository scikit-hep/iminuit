#ifndef MN_MnConfig_H_
#define MN_MnConfig_H_

// for alpha streams 
#if defined(__alpha) && !defined(linux)
#   include <standards.h>
#   ifndef __USE_STD_IOSTREAM
#   define __USE_STD_IOSTREAM
#   endif
#endif


#ifdef _MSC_VER
# pragma warning(disable:4244)  // conversion from __w64 to int
#endif

#if defined(__sun) && !defined(linux) 
#include <stdlib.h>
#endif


#endif
