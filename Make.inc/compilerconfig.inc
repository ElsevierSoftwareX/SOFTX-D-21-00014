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

# Config and compiler checks
#
ifneq (,$(wildcard $(BASEPATH)/Make.inc/$(CONFIG).inc))
include $(BASEPATH)/Make.inc/$(CONFIG).inc
else
ifneq (,$(wildcard $(CONFIG).inc))
include $(CONFIG).inc
else
$(info Cannot open config for a Makefile)
$(info Attempted to open $(BASEPATH)/Make.inc/$(CONFIG).inc or $(CONFIG).inc)
$(info NOTE: If there is no specific config file for this machine or accout, one can use CONFIG=generic)
$(info NOTE: You may also want to create new specific config file in Make.inc/ by copy-and-paste the generic.inc contents)
FATAL_ERROR=TRUE
endif
endif

include $(BASEPATH)/Make.inc/compilerflags.inc
include $(BASEPATH)/Make.inc/version.inc

CXXFLAGS_RELEASE += $(CXXOPTIMIZE_$(MACHINEID))
CXXFLAGS_COMMON += $(CXXCOMMON_$(MACHINEID))
CFLAGS_RELEASE += $(COPTIMIZE_GENERIC)
FCFLAGS_RELEASE += $(FCOPTIMIZE_GENERIC)
PROFILER_FLAGS += $(PROFILER_$(MACHINEID))

ifneq (,$(CXX_$(MACHINEID)))
ifeq ($(origin CXX),default)
CXX = $(CXX_$(MACHINEID))
endif
endif
ifneq (,$(CC_$(MACHINEID)))
ifeq ($(origin CC),default)
CC = $(CC_$(MACHINEID))
endif
endif
ifneq (,$(FC_$(MACHINEID)))
ifeq ($(origin FC),default)
FC = $(FC_$(MACHINEID))
endif
endif

ifneq (,$(MPICXX_$(MACHINEID)))
MPICXX ?= $(MPICXX_$(MACHINEID))
endif
ifneq (,$(MPICC_$(MACHINEID)))
MPICC ?= $(MPICC_$(MACHINEID))
endif
ifneq (,$(MPIFORT_$(MACHINEID)))
MPIFORT ?= $(MPIFORT_$(MACHINEID))
endif

ifeq ($(WITH_CUDA),TRUE)
ifneq (,$(NV_CXX_$(MACHINEID)))
NV_CXX ?= $(NV_CXX_$(MACHINEID))
endif
endif
ifneq (,$(CXXVERSION_$(MACHINEID)))
CXXVERSION = $(CXXVERSION_$(MACHINEID))
else
CXXVERSION = GENERIC
endif
ifeq ($(origin CXX),default)
CXX=$(CXX_GENERIC)
CXXVERSION ?= GENERIC
endif
ifeq (,$(CC))
CC = $(CC_GENERIC)
endif
ifeq (,$(FC))
FC = $(FC_GENERIC)
endif

ifeq ($(WITH_CUDA),TRUE)
DEFAULT_NV_CXX := $(shell which nvcc 2>/dev/null)
NV_CXX ?= $(DEFAULT_NV_CXX)
endif
DEFAULT_MPICXX := $(shell which mpicxx 2>/dev/null)
MPICXX ?= $(DEFAULT_MPICXX)
DEFAULT_MPICC := $(shell which mpicc 2>/dev/null)
MPICC ?= $(DEFAULT_MPICC)
DEFAULT_MPIFORT := $(shell which mpifort 2>/dev/null)
MPIFORT ?= $(DEFAULT_MPIFORT)

ifeq ($(WITH_CUDA),TRUE)
ifeq ($(NV_CXX),)
$(info Cannot find CUDA compiler (NV_CXX is empty!))
FATAL_ERROR=TRUE
endif
endif
ifeq (,$(MPICXX))
$(info Cannot find MPI C++ compiler wrapper (MPICXX is empty!))
FATAL_ERROR=TRUE
endif

ifeq ($(FATAL_ERROR),TRUE)
$(error FATAL ERROR)
endif

$(info Loaded config: $(CONFIG).inc)
$(info Configured for C++ compiler: $(shell which $(CXX)), version: $(CXXVERSION))
ifneq ($(CXXVERSION),GENERIC)
CXX_REAL_VERSION=$(shell source $(BASEPATH)/tools/dbscripts/base.inc && source $(BASEPATH)/tools/dbscripts/compchk.inc && get_cxx_cmp_version $(CXX))
MPICXX_REAL_VERSION=$(shell source $(BASEPATH)/tools/dbscripts/base.inc && source $(BASEPATH)/tools/dbscripts/compchk.inc && get_cxx_cmp_version $(MPICXX))
ifneq ($(CXXVERSION),$(CXX_REAL_VERSION))
$(info Compiler version given in CXXVERSION variable is not the same as real compiler version.)
$(info You may want to check it, update it or replace CXXVERSION value with GENERIC in actual Make.inc/ config.)
$(info NOTE: functions from tools/dbscripts/compchk.inc were used to obtain a compiler version.)
$(info NOTE: obtained real compiler version is: $(CXX_REAL_VERSION))
FATAL_ERROR=TRUE
else
ifneq ($(CXX_REAL_VERSION),$(MPICXX_REAL_VERSION))
$(info WARNING: MPICXX and CXX compiler versions are not the same!)
endif
endif
endif
