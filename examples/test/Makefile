#############################################################################
## 
##  Copyright (C) 2019-2020 Boris Krasnopolsky, Alexey Medvedev
##  Contact: xamg-test@imec.msu.ru
## 
##  This file is part of the XAMG library.
## 
##  Commercial License Usage
##  Licensees holding valid commercial XAMG licenses may use this file in
##  accordance with the terms of commercial license agreement.
##  The license terms and conditions are subject to mutual agreement
##  between Licensee and XAMG library authors signed by both parties
##  in a written form.
## 
##  GNU General Public License Usage
##  Alternatively, this file may be used under the terms of the GNU
##  General Public License, either version 3 of the License, or (at your
##  option) any later version. The license is as published by the Free 
##  Software Foundation and appearing in the file LICENSE.GPL3 included in
##  the packaging of this file. Please review the following information to
##  ensure the GNU General Public License requirements will be met:
##  https://www.gnu.org/licenses/gpl-3.0.html.
## 
#############################################################################

# XAMG Application Makefile
#
# NOTE: See Makefile usage info in Makefile.base header comments
#

TARGETS = example
BASEPATH = ../..

include ../../Makefile.base

INCLUDES += -I$(ARGSPARSER_INSTALL_PATH)
LIBS += -L$(ARGSPARSER_INSTALL_PATH) -largsparser -Wl,-rpath=$(ARGSPARSER_INSTALL_PATH)

INCLUDES += -I$(YAMLCPP_INSTALL_PATH)/include
LIBS += -L$(YAMLCPP_INSTALL_PATH)/lib -lyaml-cpp -Wl,-rpath=$(YAMLCPP_INSTALL_PATH)/lib

INCLUDES += -I$(SCOTCH_INSTALL_PATH)/include -I$(SCOTCH_INSTALL_PATH)/include_stub
LIBS += -L$(SCOTCH_INSTALL_PATH)/lib -lptscotchparmetis -lptscotch -lscotch -lptscotcherr -lz -Wl,-rpath=$(SCOTCH_INSTALL_PATH)/lib

ifeq ($(WITH_ITAC), TRUE)
CXXFLAGS += -DITAC_TRACE
endif

####

ifeq ($(WITH_SEPARATE_OBJECTS),TRUE)

# Separate object: Scotch wrapper
scotch_wrapper.o: ../common/system/graph/scotch_wrapper.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -DXAMG_SEPARATE_OBJECT -c $< -o $@

$(eval $(call add_separate_object, scotch_wrapper.o))

endif

example: example.o cmdline.o ../common/cmdline/cmdline.o $(XAMG_SEP_LIB)
	$(MPICXX) example.o cmdline.o ../common/cmdline/cmdline.o -o example $(LDFLAGS) $(LIBS)

example.o: ex_blas.h ex_hypre.h ../common/system/system.h ex_output.h ex_solver.h ex_spmv.h
cmdline.o: cmdline.h
../common/cmdline/cmdline.o: ../common/cmdline/cmdline.h
 
clean::
	rm -f example.o cmdline.o ../common/cmdline/cmdline.o example



# Typical template of XAMG Application Makefile:
#
# TARGETS = foobar
# BASEPATH = ../..   (thats where Makefile.base can be found)
#
# include $(BASEPATH)/Makefile.base
#
# foobar: foobar.o $(XAMG_SEP_LIB)
#   $(MPICXX) $^ -o foobar foobar.o $(LDFLAGS) $(LIBS)
#
# clean::
#   rm -f foobar.o foobar
#
#
# NOTE: always add $(XAMG_SEP_LIB) dependency to all individual targets
# NOTE: always add $(LDFLAGS) $(LIBS) to final link rule of every individual target
# NOTE: always use 'clean::' syntax for clean rule
