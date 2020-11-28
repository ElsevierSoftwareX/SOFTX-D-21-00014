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

#include <sstream>
#include <iostream>
#include <fstream>
#include <regex>
#include <argsparser.h>

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

#ifdef XAMG_DEBUG
#include <sys/dbg_helper.h>
#endif

#include "cmdline.h"

///////////////////

extern ID id;
extern XAMG::time_monitor monitor;
extern XAMG::perf_info perf;

///////////////////

#ifndef XAMG_NV
#define XAMG_NV 16
#endif

#ifndef FP_TYPE
#define FP_TYPE float64_t
#endif

#ifdef ITAC_TRACE
#include <VT.h>
#endif

const uint16_t NV = XAMG_NV;

#include "ex_output.h"
ex_store_output ex_output;

#include "../common/system/system.h"
#include "ex_blas.h"
#include "ex_spmv.h"
#include "ex_solver.h"
#include "ex_hypre.h"

int main(int argc, char *argv[]) {
    std::vector<std::string> solver_types = {"solver", "preconditioner", "pre_smoother",
                                             "post_smoother", "coarse_grid_solver"};
    args_parser parser(argc, argv, "-", ' ');
//    parser.add<std::string>("result").set_caption("The file to store computation results in");
#ifdef XAMG_DEBUG
    dbg_helper dbg;
    dbg.add_options_to_parser(parser);
#endif
    execution_mode_t execution_mode;
    XAMG::params::global_param_list params;
    auto res = parse_cmdline(parser, solver_types, execution_mode, params);
    switch (res) {
    case PARSE_OK:
        break;
    case PARSE_FATAL_FAILURE:
        return 1;
    case PARSE_HELP_PRINTED:
        return 0;
    }
    sleep(parser.get<int>("sleep"));
    //    auto result_filename = parser.get<std::string>("result");
    params.set_defaults();
#ifdef XAMG_DEBUG
    dbg.get_options_from_parser(argv[0]);
#endif
    XAMG::init(argc, argv, parser.get<std::string>("node_config") /*, "xamg.log"*/);
#ifdef XAMG_DEBUG
    dbg.init_output(id.gl_proc);
#endif
#ifdef ITAC_TRACE
    VT_traceoff();
#endif

    XAMG::matrix::matrix m(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector x(XAMG::mem::DISTRIBUTED), b(XAMG::mem::DISTRIBUTED);
    if (execution_mode != blas_test) {
        make_system<FP_TYPE>(parser, m, x, b, parser.get<bool>("graph_reordering"));
        if (!b.if_initialized)
            XAMG::blas::set_const<FP_TYPE, NV>(b, 1.0);
        if (!x.if_initialized)
            XAMG::blas::set_const<FP_TYPE, NV>(x, 0.0);
        //        XAMG::out.norm<FP_TYPE, NV>(x, "x0");
        //        XAMG::out.norm<FP_TYPE, NV>(b, "b0");
    }
    ex_output.init();
    if (execution_mode == blas_test) { // blas functions
        ex_blas_test<FP_TYPE>();
    }
    if (execution_mode == solver_test) { // solver
        ex_solver_test<FP_TYPE>(m, x, b, params);
    }
    if (execution_mode == spmv_test) { // spmv
        ex_spmv_test<FP_TYPE>(m, b, x);
    }
    if (execution_mode == hypre_test) {
        ex_hypre_test(m, x, b, params);
    }

    XAMG::finalize();
    // if (id.master_process())
    //     ex_output.dump(result_filename);
}
