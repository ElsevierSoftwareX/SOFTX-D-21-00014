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

# Common flags
CFLAGS_DEBUG   = -O0 -g
CFLAGS_COMMON  = -fPIC -Wall -Wno-unused-variable -Wno-unused-but-set-variable
FCFLAGS_DEBUG   = -O0 -g
FCFLAGS_COMMON  = -fPIC -Wall -Wno-unused-variable -Wno-unused-but-set-variable

CXXFLAGS_DEBUG   = -O0 -g -DXAMG_DEBUG -gdwarf-2
CXXFLAGS_DEBUG_EXTRA = -gdwarf-2
CXXFLAGS_COMMON  = -std=c++11 
CXXFLAGS_COMMON  += -fPIC 
CXXFLAGS_COMMON  += -Wall -Wno-unused-variable -Wno-unused-but-set-variable
NV_CXXFLAGS += -std=c++11

# Generic optimization flags
CXX_GENERIC = c++
CC_GENERIC = cc
FC_GENERIC = f95
MPICXX_GENERIC = mpicxx
MPIC_GENERIC = mpicc
MPIFORT_GENERIC = mpifort
CXXVERSION_GENERIC = GENERIC
CXXOPTIMIZE_GENERIC = -O3 -DNDEBUG
COPTIMIZE_GENERIC = -O3 -DNDEBUG
FCOPTIMIZE_GENERIC = -O3 -DNDEBUG
PROFILER_GENERIC = -pg

PROFILER_GCC = -pg
PROFILER_ICC = -p

# Lomonosov-2, icc 19 common
CXX_LOM2_ICC19 = icpc
CXXVERSION_LOM2_ICC19 = 19.0.5.281
CXXOPTIMIZE_LOM2_ICC19 = -ip -O3 -no-prec-div -static -fp-model fast=2 -qopt-report=5 -qopt-report-phase=loop,vec -DNDEBUG

# Lomonosov-2, icc 19, nodes in compute queue
CXX_LOM2_ICC19_COMPUTEQ = $(CXX_LOM2_ICC19)
CXXVERSION_LOM2_ICC19_COMPUTEQ = $(CXXVERSION_LOM2_ICC19)
CXXOPTIMIZE_LOM2_ICC19_COMPUTEQ = $(CXXOPTIMIZE_LOM2_ICC19) -axCORE-AVX2
PROFILER_LOM2_ICC19_COMPUTEQ = $(PROFILER_ICC)

# Lomonosov-2, icc 19, nodes in pascal queue
CXX_LOM2_ICC19_PASCALQ = $(CXX_LOM2_ICC19)
CXXVERSION_LOM2_ICC19_PASCALQ = $(CXXVERSION_LOM2_ICC19)
CXXOPTIMIZE_LOM2_ICC19_PASCALQ = $(CXXOPTIMIZE_LOM2_ICC19) -axCORE-AVX512
PROFILER_LOM2_ICC19_PASCALQ = $(PROFILER_ICC)

# KIAE4
CXX_KIAE4 = g++
CXXVERSION_KIAE4 = 8.3.1
CXXOPTIMIZE_KIAE4 = -O3 -ffast-math -funroll-loops -ftree-vectorize -march=haswell -DNDEBUG
PROFILER_KIAE4 = $(PROFILER_GCC)
