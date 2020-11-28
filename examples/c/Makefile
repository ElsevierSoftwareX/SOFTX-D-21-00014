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

# Makefile
#
# NOTE: Local Makefile to compile C API usage example.
# CC, MPICC, basic C compilation flags are specified in Make.inc/compilerflags.inc

TARGETS = example
BASEPATH = ../..

include $(BASEPATH)/Makefile.base

XAMG_C_API = $(BASEPATH)/api/c
XAMG_USER_FLAGS ?= "-DXAMG_NV=2"

LIBS += -L$(XAMG_C_API) -Wl,-rpath=$(XAMG_C_API) -lxamg_c_api

xamg_c_api_lib:
	cd $(XAMG_C_API) && $(MAKE) clean && $(MAKE) XAMG_USER_FLAGS=$(XAMG_USER_FLAGS)

example: example.o $(XAMG_SEP_LIB)
	$(MAKE) xamg_c_api_lib
	$(MPICC) $^ -o example $(LDFLAGS) $(LIBS)

clean::
	rm -f example.o example
