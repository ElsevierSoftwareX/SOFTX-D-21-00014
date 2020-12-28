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

#pragma once

#include "part/part.h"

template <typename F>
void tst_blas_test() {
    double t1, t2;
    uint64_t iters;

    uint64_t vec_size = 100;
    uint64_t block_size = vec_size / id.gl_nprocs;
    if (id.gl_proc == id.gl_nprocs - 1)
        block_size = vec_size - (id.gl_nprocs - 1) * block_size;

    std::shared_ptr<XAMG::part::part> part = XAMG::part::get_shared_part();
    part->get_part(block_size);

    XAMG::vector::vector x(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector y(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector z(XAMG::mem::DISTRIBUTED);

    x.alloc<F>(part->numa_layer.block_size[id.nd_numa], NV);
    y.alloc<F>(part->numa_layer.block_size[id.nd_numa], NV);
    z.alloc<F>(part->numa_layer.block_size[id.nd_numa], NV);
    x.set_part(part);
    y.set_part(part);
    z.set_part(part);
    XAMG::blas::set_const<F, NV>(x, 1.01);
    XAMG::blas::set_const<F, NV>(y, 0.95);
    XAMG::blas::set_const<F, NV>(z, 1.0);

    XAMG::vector::vector a1, a2, a3;
    a1.alloc<F>(1, NV);
    a2.alloc<F>(1, NV);
    a3.alloc<F>(1, NV);
    XAMG::blas::set_const<F, NV>(a1, 1.0);
    XAMG::blas::set_const<F, NV>(a2, 0.99);
    XAMG::blas::set_const<F, NV>(a3, 0.01);

    //////////

    //// warm-up
    // for (uint64_t i = 0; i < 10; i++) {
    //     XAMG::blas::axpby<F, NV>(a1, x, a2, y);
    //     XAMG::blas::axpby<F, NV>(a1, x, a2, z);
    //     XAMG::blas::axpby<F, NV>(a2, y, a3, z);
    // }

    XAMG::mpi::barrier();
    t1 = XAMG::sys::timer();

    for (uint64_t i = 0; i < 10; i++) {
        XAMG::blas::axpby<F, NV>(a1, x, a2, y);
        // XAMG::blas::axpby<F, NV>(a1, x, a2, z);
        // XAMG::blas::axpby<F, NV>(a2, y, a3, z);
    }

    XAMG::mpi::barrier();
    t2 = XAMG::sys::timer();

    // iters = (int)(5.0 / ((t2-t1) / 10.0));
    // MPI_Bcast (&iters, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    // XAMG::mpi::bcast<uint64_t>(&iters, 1, 0);

    XAMG::vector::vector res;
    res.alloc<F>(1, NV);

    XAMG::blas::vector_norm<F, NV>(y, XAMG::L2_norm, res, true);

    // XAMG::out << "Vector size : " << vec_size << " : iters " << iters << std::endl;
    XAMG::out.vector<F>(res, " res ");
}
