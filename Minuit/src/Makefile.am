## Process this file with automake to produce Makefile.in .
#
# Author: Paul_Kunz@slac.stanford.edu
#

# The following is set; otherwise it would follow GNU conventions.
AUTOMAKE_OPTIONS = foreign

# The convenience library to be built.
lib_LTLIBRARIES = liblcg_Minuit.la

liblcg_Minuit_la_SOURCES =				\
         AnalyticalGradientCalculator.cpp	\
         BasicMinimumError.cpp			\
         CombinedMinimumBuilder.cpp		\
         DavidonErrorUpdator.cpp		\
         FumiliBuilder.cpp                      \
         FumiliErrorUpdator.cpp                 \
         FumiliGradientCalculator.cpp           \
         FumiliMinimizer.cpp                    \
         FumiliStandardChi2FCN.cpp              \
	 FumiliStandardMaximumLikelihoodFCN.cpp \
         HessianGradientCalculator.cpp		\
         InitialGradientCalculator.cpp		\
         LaEigenValues.cpp			\
         LaInnerProduct.cpp			\
         LaInverse.cpp				\
         LaOuterProduct.cpp			\
         LaSumOfElements.cpp			\
         LaVtMVSimilarity.cpp			\
         MnApplication.cpp			\
         MnContours.cpp				\
         MnCovarianceSqueeze.cpp		\
         MnEigen.cpp				\
         MnFcn.cpp				\
         MnFunctionCross.cpp			\
         MnFumiliMinimize.cpp			\
         MnGlobalCorrelationCoeff.cpp		\
         MnHesse.cpp				\
         MnLineSearch.cpp			\
         MnMachinePrecision.cpp			\
         MnMinos.cpp				\
         MnParabolaFactory.cpp			\
         MnParameterScan.cpp			\
         MnPlot.cpp				\
         MnPosDef.cpp				\
         MnPrint.cpp				\
         MnScan.cpp				\
         MnSeedGenerator.cpp			\
         MnStrategy.cpp				\
         MnTiny.cpp				\
         MnUserFcn.cpp				\
         MnUserParameterState.cpp		\
         MnUserParameters.cpp			\
         MnUserTransformation.cpp		\
         ModularFunctionMinimizer.cpp		\
         NegativeG2LineSearch.cpp		\
         Numerical2PGradientCalculator.cpp	\
         ParametricFunction.cpp                 \
         ScanBuilder.cpp			\
         SimplexBuilder.cpp			\
         SimplexParameters.cpp			\
         SimplexSeedGenerator.cpp		\
         SinParameterTransformation.cpp		\
         SqrtLowParameterTransformation.cpp	\
         SqrtUpParameterTransformation.cpp	\
         VariableMetricBuilder.cpp		\
         VariableMetricEDMEstimator.cpp		\
         mnbins.cpp				\
         mndasum.cpp				\
         mndaxpy.cpp				\
         mnddot.cpp				\
         mndscal.cpp				\
         mndspmv.cpp				\
         mndspr.cpp				\
         mnteigen.cpp				\
         mnlsame.cpp				\
	 mntplot.cpp				\
	 mnvert.cpp				\
         mnxerbla.cpp                        

INCLUDES =					\
	-I$(top_srcdir)

# For platforms where automake dependency generation doesn't work.
depend: $(liblcg_Minuit_la_SOURCES)
	$(top_srcdir)/config/makedepend $(DEFS) $(INCLUDES) $(CPPFLAGS) $?

# An extra clean target for the Sun CC compiler.
clean-local:
	-rm -rf SunWS_cache
