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

#include "../common/generator/generator.h"

#ifndef XAMG_NV
#define XAMG_NV 16
#endif

const uint16_t NV = XAMG_NV;

int main(int argc, char *argv[]) {

    // XAMG::init(argc, argv/*, "ncores=64;nnumas=2;ngpus=2", "xamg.log"*/);
    // XAMG::init(argc, argv/*, "0xffffffff@0;0xffffffff@1", "xamg.log"*/);
    XAMG::init(argc, argv /*, "", "xamg.log"*/);

    std::string method = "BiCGStab";
    XAMG::params::global_param_list params;
    params.add("solver", {"method", method});
    params.set_defaults();

    XAMG::matrix::matrix m(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector x(XAMG::mem::DISTRIBUTED), b(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector x0(XAMG::mem::LOCAL), b0(XAMG::mem::LOCAL);

    XAMG::matrix::csr_matrix<float64_t, uint32_t, uint32_t, uint32_t, uint32_t> mat_csr;
    generate_system<float64_t, uint32_t, uint32_t, uint32_t, uint32_t, NV>(mat_csr, x0, b0, 10, 10,
                                                                           10);

    std::shared_ptr<XAMG::part::part> part = XAMG::part::get_shared_part();
    part->get_part(mat_csr.nrows);
    m.set_part(part);

    m.construct(mat_csr);

    x.alloc<float64_t>(m.row_part->numa_layer.block_size[id.nd_numa], NV);
    b.alloc<float64_t>(m.row_part->numa_layer.block_size[id.nd_numa], NV);
    x.set_part(part);
    b.set_part(part);
    XAMG::blas::copy<float64_t, NV>(x0, x);
    XAMG::blas::copy<float64_t, NV>(b0, b);

    auto sol = XAMG::solver::construct_solver_hierarchy<float64_t, NV>(params, m, x, b);

    XAMG::mpi::barrier();
    double t1 = XAMG::io::timer();

    sol->solve();

    XAMG::mpi::barrier();
    double t2 = XAMG::io::timer();

    XAMG::out << "Time: " << t2 - t1 << std::endl;
    XAMG::finalize();
}
