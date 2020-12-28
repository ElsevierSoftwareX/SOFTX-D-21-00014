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
void ex_solver_test(XAMG::matrix::matrix &m, XAMG::vector::vector &x, XAMG::vector::vector &y,
                    const XAMG::params::global_param_list &params) {
    double t1, t2;

    auto sol = XAMG::solver::construct_solver_hierarchy<F, NV>(params, m, x, y);

    XAMG::out.norm<F, NV>(x, "x0");
    XAMG::out.norm<F, NV>(y, "y0");

    XAMG::mpi::barrier();
    t1 = XAMG::sys::timer();

    sol->solve();

    XAMG::mpi::barrier();
    t2 = XAMG::sys::timer();

    std::string method;
    sol->param_list.get_value("method", method);

    XAMG::out << XAMG::SUMMARY << "Solver: " << method << "\tMatrix size: " << sol->A.info.nrows
              << "\tMatrix nonzeros: " << sol->A.info.nonzeros << std::endl;
    XAMG::out << XAMG::SUMMARY << "Nvecs: " << NV << "   nprocs: " << id.gl_nprocs
              << "   nnumas: " << id.nd_nnumas << "   ncores: " << id.nm_ncores
              << "   Solver time: " << t2 - t1 << " sec \tIters: " << sol->stats.iters << std::endl;

    for (uint16_t nv = 0; nv < NV; ++nv) {
        XAMG::out << XAMG::SUMMARY << "Vec: " << nv
                  << "\tConverged: " << sol->stats.if_converged[nv]
                  << "\tAbs.residual: " << sol->stats.abs_res[nv] << std::endl;
    }

    XAMG::out.norm<F, NV>(x, "X");
}
