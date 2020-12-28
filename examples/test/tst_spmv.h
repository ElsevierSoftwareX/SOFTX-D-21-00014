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

template <typename F>
void tst_spmv_test(XAMG::matrix::matrix &m, const XAMG::vector::vector &x,
                   XAMG::vector::vector &y) {

    double t1, t2;
    uint32_t iters = 10;

    XAMG::mpi::barrier();

    for (uint16_t it = 0; it < 3; ++it) {
        XAMG::blas::set_const<F, NV>(y, 0.0, true);

        XAMG::mpi::barrier();
        t1 = XAMG::sys::timer();

        for (uint32_t i = 0; i < iters; i++) {
            XAMG::blas2::Axpy<F, NV>(m, x, y, NV);
        }

        XAMG::mpi::barrier();
        t2 = XAMG::sys::timer();

        XAMG::out << XAMG::SUMMARY << "SUMMARY:: "
                  << "SpMV: \tMatrix size: " << m.info.nrows
                  << "\tMatrix nonzeros: " << m.info.nonzeros << std::endl;
        XAMG::out << XAMG::SUMMARY << "Nvecs: " << NV << "   nprocs: " << id.gl_nprocs
                  << "   nnumas: " << id.nd_nnumas << "   ncores: " << id.nm_ncores
                  << "   SpMV time: " << t2 - t1 << " sec \tIters: " << iters << std::endl;

        //        XAMG::out.format("Exec. time: %.6f sec \tIters: %d\n", t2-t1, iters);
    }

    XAMG::out.norm<F, NV>(y, " x = A * b ");
}
