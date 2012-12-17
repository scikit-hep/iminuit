## Process this file with automake to produce Makefile.in .
#
# Author: Paul_Kunz@slac.stanford.edu
#

# The following is set; otherwise, it would follow GNU conventions.
# AUTOMAKE_OPTIONS = foreign

## Subsitution from pfkeb_cxx_lib_path
CXX_LIB_PATH = @CXX_LIB_PATH@

check_PROGRAMS =	\
	test_Minuit_DemoGaussSim   \
	test_Minuit_DemoFumili   \
	test_Minuit_PaulTest    \
	test_Minuit_PaulTest2  \
	test_Minuit_PaulTest3  \
	test_Minuit_PaulTest4  \
	test_Minuit_ReneTest  

test_Minuit_DemoGaussSim_SOURCES =	\
	GaussFunction.h \
	GaussRandomGen.h \
	FlatRandomGen.h \
	GaussFcn.h	\
	GaussFcn.cpp	\
	GaussDataGen.h	\
	GaussDataGen.cpp	\
	DemoGaussSim.cpp			

test_Minuit_DemoFumili_SOURCES =	\
	GaussianModelFunction.h \
	GaussDataGen.h	\
	GaussDataGen.cpp	\
	DemoFumili.cpp			

test_Minuit_PaulTest_SOURCES =	PaulTest.cpp 

test_Minuit_PaulTest2_SOURCES =	PaulTest2.cpp 

test_Minuit_PaulTest3_SOURCES =	        \
		      GaussFcn2.h       \
                      GaussFcn2.cpp     \
                      PaulTest3.cpp

test_Minuit_PaulTest4_SOURCES =	PaulTest4.cpp 

test_Minuit_ReneTest_SOURCES =	ReneTest.cpp 

INCLUDES =					\
	-I$(top_srcdir)	

test_Minuit_DemoGaussSim_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_DemoGaussSim_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit_DemoFumili_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_DemoFumili_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit_PaulTest_LDADD =	 \
	GaussFcn.o			\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_PaulTest_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit_PaulTest2_LDADD =				\
	GaussFcn.o			\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_PaulTest2_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit_PaulTest3_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_PaulTest3_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit_PaulTest4_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_PaulTest4_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit_ReneTest_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_ReneTest_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)



AllSOURCES =				        \
	$(test_Minuit_DemoGaussSim_SOURCES)     \
	$(test_Minuit_DemoFumili_SOURCES)       \
	$(test_Minuit_PaulTest_SOURCES)		\
	$(test_Minuit_PaulTest2_SOURCES)        \
	$(test_Minuit_PaulTest3_SOURCES)	\
	$(test_Minuit_PaulTest4_SOURCES)	\
	$(test_Minuit_ReneTest_SOURCES)		

EXTRA_DIST = BuildFile paul.txt paul2.txt paul3.txt paul4.txt

# For platforms where automake dependency generation doesn't work.
depend: $(AllSOURCES)
	$(top_srcdir)/config/makedepend $(DEFS) $(INCLUDES) $(CPPFLAGS) $?

clean-local:
	-rm -rf *.C

