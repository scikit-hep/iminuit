## Process this file with automake to produce Makefile.in .
#
# Author: Paul_Kunz@slac.stanford.edu
#

# The following is set; otherwise, it would follow GNU conventions.
# AUTOMAKE_OPTIONS = foreign

## Subsitution from pfkeb_cxx_lib_path
CXX_LIB_PATH = @CXX_LIB_PATH@

check_PROGRAMS =	\
	test_Minuit_Quad4FMain	\
	test_Minuit_Quad8FMain	\
	test_Minuit_Quad12FMain	\
	test_Minuit_Quad1FMain

test_Minuit_Quad4FMain_SOURCES =	\
	Quad4F.h	\
	Quad4FMain.cpp			

test_Minuit_Quad1FMain_SOURCES =	\
	Quad1F.h	\
	Quad1FMain.cpp			

test_Minuit_Quad8FMain_SOURCES =	\
	Quad8F.h	\
	Quad8FMain.cpp			

test_Minuit_Quad12FMain_SOURCES =	\
	Quad12F.h	\
	Quad12FMain.cpp			

AllSOURCES =				\
	$(test_Minuit_Quad4FMain_SOURCES)		\
	$(test_Minuit_Quad8FMain_SOURCES)		\
	$(test_Minuit_Quad12FMain_SOURCES)		\
	$(test_Minuit_Quad1FMain_SOURCES)		

INCLUDES =				\
	-I$(top_srcdir)	


test_Minuit_Quad4FMain_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_Quad1FMain_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la


test_Minuit_Quad8FMain_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la

test_Minuit_Quad12FMain_LDADD =				\
	$(top_builddir)/src/liblcg_Minuit.la


test_Minuit_Quad4FMain_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit_Quad1FMain_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)


test_Minuit_Quad8FMain_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit_Quad12FMain_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)


EXTRA_DIST = BuildFile

# For platforms where automake dependency generation doesn't work.
depend: $(AllSOURCES)
	$(top_srcdir)/config/makedepend $(DEFS) $(INCLUDES) $(CPPFLAGS) $?

clean-local:
	-rm -rf *.C

