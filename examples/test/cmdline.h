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

#include "param/params.h"
#include "../common/cmdline/cmdline.h"

enum execution_mode_t { blas_test, spmv_test, solver_test, hypre_test, none };

static inline execution_mode_t str_to_execution_mode(const std::string &str) {
    if (str == "spmv") {
        return execution_mode_t::spmv_test;
    } else if (str == "solver") {
        return execution_mode_t::solver_test;
    } else if (str == "blas") {
        return execution_mode_t::blas_test;
    } else if (str == "hypre") {
        return execution_mode_t::hypre_test;
    } else {
        return execution_mode_t::none;
    }
}

#define MODE_CAPTION "<solver|hypre|spmv|blas>"
#define MODE_DESCR ""

#define GRAPH_REORDERING_CAPTION                                                                   \
    "- Flag activating usage of graph partitioning library. See detailed help."
#define GRAPH_REORDERING_DESCR ""

parse_result_t parse_cmdline(args_parser &parser, const std::vector<std::string> &solver_types,
                             execution_mode_t &mode, XAMG::params::global_param_list &params);
