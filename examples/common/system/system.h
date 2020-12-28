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

#include "../generator/generator.h"
#include "reorder.h"

template <typename F>
void make_system(args_parser &parser, XAMG::matrix::matrix &m, XAMG::vector::vector &x,
                 XAMG::vector::vector &b, const bool graph_reordering = false) {
    auto cmdline_matrix = parser.get<std::string>("matrix");
    using matrix_t = XAMG::matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t>;
    auto sh_mat_csr = std::make_shared<matrix_t>();
    auto sh_x0 = std::make_shared<XAMG::vector::vector>();
    auto sh_b0 = std::make_shared<XAMG::vector::vector>();

    if (cmdline_matrix != "generate") {
        XAMG::io::read_system<matrix_t, NV>(*sh_mat_csr, *sh_x0, *sh_b0, cmdline_matrix);
    } else {
        std::map<std::string, std::string> cmdline_generator_params;
        parser.get("generator_params", cmdline_generator_params);
        generator_params_t generator_params;
        parse_generator_params(cmdline_generator_params, generator_params);
        generate_system<F, uint32_t, uint32_t, uint32_t, uint32_t, NV>(*sh_mat_csr, *sh_x0, *sh_b0,
                                                                       generator_params);
    }

    //  use graph methods to reorder the matrix
    if (graph_reordering) {
        reorder_system(sh_mat_csr, sh_x0, sh_b0);
        assert(sh_b0->size == sh_mat_csr->nrows);
        XAMG::out << "Partitioning completed...\n";
    }

    auto part = XAMG::part::make_partitioner(sh_mat_csr->nrows);
    XAMG::matrix::construct_distributed<matrix_t>(part, *sh_mat_csr, m);
    XAMG::vector::construct_distributed<F, NV>(part, *sh_x0, x);
    XAMG::vector::construct_distributed<F, NV>(part, *sh_b0, b);

    //    x.print<float64_t>("X vec");
    //    b.print<float64_t>("B vec");

    //    XAMG::out << XAMG::ALLRANKS << "Ex_matrix completed...\n";
    //    XAMG::io::sync();

    return;
}
