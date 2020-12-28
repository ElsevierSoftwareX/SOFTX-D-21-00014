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

#include <xamg/xamg_headers.h>
#include <xamg/xamg_types.h>
#include <xamg/init.h>
#include <xamg/blas/blas.h>
#include <xamg/blas2/blas2.h>
#include <xamg/solvers/solver.h>

#include "../common/generator/generator.h"

int main(int argc, char *argv[]) {
    using FP = double;
    using INT = uint32_t;
    using matrix_t = XAMG::matrix::csr_matrix<FP, INT, INT, INT, INT>;
    constexpr uint16_t NV = 8;

    XAMG::init(argc, argv, "nnumas=1:ncores=1");

    XAMG::matrix::csr_matrix<FP, INT, INT, INT, INT> csr_file_mtx;
    XAMG::vector::vector csr_file_x, csr_file_b;

    generator_params_t p;
    p.nx = p.ny = p.nz = 70;
    generate_system<FP, INT, INT, INT, INT, NV>(csr_file_mtx, csr_file_x, csr_file_b, p);

    XAMG::matrix::matrix m(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector x(XAMG::mem::DISTRIBUTED), b(XAMG::mem::DISTRIBUTED);
    auto part = XAMG::part::make_partitioner(csr_file_mtx.nrows);
    XAMG::matrix::construct_distributed<matrix_t>(part, csr_file_mtx, m);
    XAMG::vector::construct_distributed<FP, NV>(part, csr_file_x, x);
    XAMG::vector::construct_distributed<FP, NV>(part, csr_file_b, b);

    XAMG::params::global_param_list params;
    params.add("solver", {"method", "PBiCGStab"});
    params.add_map("solver", {{"max_iters", "20"}});
    params.add_map("preconditioner", {{"method", "MultiGrid"},
                                      {"max_iters", "1"},
                                      {"mg_agg_num_levels", "2"},
                                      {"mg_coarse_matrix_size", "500"},
                                      {"mg_num_paths", "2"}});
    params.add_map("pre_smoother", {{"method", "Chebyshev"}, {"polynomial_order", "2"}});
    params.add_map("post_smoother", {{"method", "Chebyshev"}, {"polynomial_order", "2"}});
    params.set_defaults();

    auto solver = XAMG::solver::construct_solver_hierarchy<FP, NV>(params, m, x, b);

    solver->solve();
    for (auto &x : solver->stats.abs_res)
        std::cout << ">> " << x << std::endl;

    XAMG::finalize();
    return 0;
}
