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
                    XAMG::params::global_param_list &params) {
    params.print();

    // solving SLAE y = A*x
    auto sol = XAMG::solver::construct_solver_hierarchy<F, NV>(params, m, x, y);
    if (sol->precond == nullptr)
        sol->matrix_info();
    else
        sol->precond->matrix_info();

    /*
    // --- rel_tolerance reset example
    params.change_value<float32_t>("solver", "rel_tolerance", 1e-12);
    if (params.find_if<std::string>({{"method", "MultiGrid"}})) {
        params.forced_change_value<uint16_t>("preconditioner", "mg_cycle", 2);
    }
    sol->renew_params(params);

    // --- override mg_num_levels
    std::string solver_type;
    if (params.find_if<std::string>({{"method", "MultiGrid"}}, solver_type)) {
        params.forced_change_value<uint16_t>(solver_type, "mg_max_levels", 2);

        params.change_value_onlayer<uint16_t>("pre_smoother", "max_iters", 3, 3);
    }
    params.change_value_onlayer<uint16_t>("pre_smoother", "max_iters", 12, 3);
    sol->renew_params(params);
    params.print();
	*/
    // --- getting minmax diapason for a parameter example
    //std::pair<uint16_t, uint16_t> minmax;
    //if (params.get("solver").get_minmax<uint16_t>("mg_coarsening_type", minmax)) {
    //    std::cout << ">> *** mg_coarsening_type: {" << minmax.first << "," << minmax.second << "}" << std::endl;
    //}
    //if (params.get("solver").get_minmax<uint16_t>("mg_interpolation_type", minmax)) {
    //    std::cout << ">> *** mg_interpolation_type: {" << minmax.first << "," << minmax.second << "}" << std::endl;
    //}
    // ---

    double t1, t2;
    for (uint16_t it = 0; it < 3; ++it) {

        perf.reset();

        std::vector<F> a;
        for (size_t m = 0; m < NV; ++m)
            a.push_back(m);
        XAMG::blas::set_const<F, NV>(x, a);
        //        XAMG::blas::set_rand<F, NV>(x, false);
        XAMG::out.norm<F, NV>(x, "x0");
        XAMG::out.norm<F, NV>(y, "y0");

        monitor.reset();
        monitor.enable();
#ifdef ITAC_TRACE
        VT_traceon();
#endif
        XAMG::mpi::barrier();
        t1 = XAMG::io::timer();

        sol->solve();

        XAMG::mpi::barrier();
        t2 = XAMG::io::timer();
#ifdef ITAC_TRACE
        VT_traceoff();
#endif
        monitor.disable();

        perf.print();

        std::string method;
        sol->param_list.get_value("method", method);

        XAMG::out << XAMG::SUMMARY << "Solver: " << method << "\tMatrix size: " << sol->A.info.nrows
                  << "\tMatrix nonzeros: " << sol->A.info.nonzeros << std::endl;
        XAMG::out << XAMG::SUMMARY << "Nvecs: " << NV << "   nprocs: " << id.gl_nprocs
                  << "   nnumas: " << id.nd_nnumas << "   ncores: " << id.nm_ncores
                  << "   Solver time: " << t2 - t1 << " sec \tIters: " << sol->stats.iters
                  << std::endl;
        for (uint16_t nv = 0; nv < NV; ++nv) {
            XAMG::out << XAMG::SUMMARY << "Vec: " << nv
                      << "\tConverged: " << sol->stats.if_converged[nv]
                      << "\tAbs.residual: " << sol->stats.abs_res[nv] << std::endl;
        }

        monitor.print();
    }

    XAMG::out.norm<F, NV>(x, "X");
    ex_output.store_xamg_vector_norm<F, NV>("solver", "X_norm", x);

    XAMG::vector::vector res;
    res.alloc<F>(1, NV);

    XAMG::vector::vector temp(x);
    XAMG::blas::set_const<F, NV>(temp, 0.0, true);
    sol->get_residual(x, temp, res);
    XAMG::blas::pwr<F, NV>(res, 0.5);
    XAMG::mpi::barrier(XAMG::mpi::INTRA_NODE);

    if (id.master_process())
        XAMG::out.vector<F>(res, "||b - A * x||");

    ex_output.store_item("solver", "iters", sol->stats.iters);
    ex_output.store_item("solver", "converged", sol->stats.if_converged);
    ex_output.store_item("solver", "abs_residual", sol->stats.abs_res);
    ex_output.store_xamg_vector<F>("solver", "residual", res);

    ex_output.store_item("info", "nrows", sol->A.info.nrows);
    ex_output.store_item("info", "nonzeros", sol->A.info.nonzeros);
    ex_output.store_item("info", "NV", NV);
    ex_output.store_item("info", "nprocs", id.gl_nprocs);
}
