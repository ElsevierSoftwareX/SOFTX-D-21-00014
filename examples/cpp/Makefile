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
#

TARGETS = all
BASEPATH = ../..

include $(BASEPATH)/Makefile.base

minimal: minimal.o $(XAMG_SEP_LIB)
	$(MPICXX) $^ -o minimal $(LDFLAGS) $(LIBS)

example: example.o $(XAMG_SEP_LIB)
	$(MPICXX) $^ -o example $(LDFLAGS) $(LIBS)

example.o: ex_solver.h

all:
	$(MAKE) example
	$(MAKE) minimal

clean::
	rm -f minimal.o example.o example minimal


# Typical template of XAMG Application Makefile:
#
# TARGETS = foobar
# BASEPATH = ../..   (thats where Makefile.base can be found)
#
# include $(BASEPATH)/Makefile.base
#
# foobar: foobar.o $(XAMG_SEP_LIB)
# 	$(MPICXX) $^ -o foobar foobar.o $(LDFLAGS) $(LIBS)
#
# clean::
#   rm -f foobar.o foobar
#
#
# NOTE: always add $(XAMG_SEP_LIB) dependency to all individual targets
# NOTE: always add $(LDFLAGS) $(LIBS) to final link rule of every individual target
# NOTE: always use 'clean::' syntax for clean rule
#
