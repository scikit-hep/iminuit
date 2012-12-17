/* xerbla.f -- translated by f2c (version 20010320).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#if defined(__alpha) 
#define __USE_STD_IOSTREAM
#endif

#include <iostream>

/* Table of constant values */

// static integer c__1 = 1;

int mnxerbla(const char* srname, int info) {
    /* Format strings */
//     static char fmt_9999[] = "(\002 ** On entry to \002,a6,\002 parameter nu\// mber \002,i2,\002 had \002,\002an illegal value\002)";
  
/*  -- LAPACK auxiliary routine (version 3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     Courant Institute, Argonne National Lab, and Rice University */
/*     September 30, 1994 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  XERBLA  is an error handler for the LAPACK routines. */
/*  It is called by an LAPACK routine if an input parameter has an */
/*  invalid value.  A message is printed and execution stops. */

/*  Installers may consider modifying the STOP statement in order to */
/*  call system-specific exception-handling facilities. */

/*  Arguments */
/*  ========= */

/*  SRNAME  (input) CHARACTER*6 */
/*          The name of the routine which called XERBLA. */

/*  INFO    (input) INTEGER */
/*          The position of the invalid parameter in the parameter list */
/*          of the calling routine. */

/* ===================================================================== */

/*     .. Executable Statements .. */

  std::cout<<" ** On entry to "<<srname<<" parameter number "<<info<<" had an illegal value"<<std::endl;

/*     End of XERBLA */

    return 0;
} /* xerbla_ */

