/****************************************************************************
** 
**  Copyright (C) 2019-2020 Boris Krasnopolsky, Alexey Medvedev
**  Contact: xamg-test@imec.msu.ru
** 
**  This file is part of the XAMG library.
** 
**  Commercial License Usage
**  Licensees holding valid commercial XAMG licenses may use this file in
**  accordance with the terms of commercial license agreement.
**  The license terms and conditions are subject to mutual agreement
**  between Licensee and XAMG library authors signed by both parties
**  in a written form.
** 
**  GNU General Public License Usage
**  Alternatively, this file may be used under the terms of the GNU
**  General Public License, either version 3 of the License, or (at your
**  option) any later version. The license is as published by the Free 
**  Software Foundation and appearing in the file LICENSE.GPL3 included in
**  the packaging of this file. Please review the following information to
**  ensure the GNU General Public License requirements will be met:
**  https://www.gnu.org/licenses/gpl-3.0.html.
** 
****************************************************************************/

#include <iostream>

#include <xamg_headers.h>
#include <xamg_types.h>

#include <init.h>

#include <primitives/vector/vector.h>
#include <primitives/matrix/matrix.h>

#include <blas/blas.h>
#include <blas2/blas2.h>
#include <solvers/solver.h>

#include <io/io.h>
#include <part/part.h>

#include <param/params.h>

///////////////////

extern ID id;
extern XAMG::perf_info perf;

///////////////////

#ifndef XAMG_NV
#define XAMG_NV 16
#endif

#define FP_TYPE float64_t

const uint16_t NV = XAMG_NV;

#include "ex_solver.h"
#include "../common/generator/generator.h"

int main(int argc, char *argv[]) {
    // XAMG::init(argc, argv/*, "ncores=64;nnumas=2;ngpus=2", "xamg.log"*/);
    // XAMG::init(argc, argv/*, "0xffffffff@0;0xffffffff@1", "xamg.log"*/);
    XAMG::init(argc, argv /*, "", "xamg.log"*/);

    XAMG::params::global_param_list params;
    params.add("solver", {"method", "BiCGStab"});
    params.set_defaults();

    XAMG::matrix::matrix m(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector x(XAMG::mem::DISTRIBUTED), b(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector x0(XAMG::mem::LOCAL), b0(XAMG::mem::LOCAL);

    XAMG::matrix::csr_matrix<FP_TYPE, uint32_t, uint32_t, uint32_t, uint32_t> mat_csr;
    generate_system<FP_TYPE, uint32_t, uint32_t, uint32_t, uint32_t, NV>(mat_csr, x0, b0, 10, 10,
                                                                         10);
    //    XAMG::out << "Data initialization completed...\n";

    std::shared_ptr<XAMG::part::part> part = XAMG::part::get_shared_part();
    part->get_part(mat_csr.nrows);
    m.set_part(part);

    m.construct(mat_csr);

    x.alloc<FP_TYPE>(m.row_part->numa_layer.block_size[id.nd_numa], NV);
    b.alloc<FP_TYPE>(m.row_part->numa_layer.block_size[id.nd_numa], NV);
    x.set_part(part);
    b.set_part(part);
    XAMG::blas::copy<FP_TYPE, NV>(x0, x);
    XAMG::blas::copy<FP_TYPE, NV>(b0, b);

    ex_solver_test<FP_TYPE>(m, x, b, params);

    XAMG::finalize();
}
