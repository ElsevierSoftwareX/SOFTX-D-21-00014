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

namespace XAMG {
namespace hypre {

struct hypre_base {
    //  square matrix expected
    std::shared_ptr<part::part> part;
    std::vector<int> row_indx;

    HYPRE_IJMatrix hA;
    HYPRE_IJVector hb, hx;

    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_Solver solver;
    HYPRE_Solver precond;

    bool mg_precond_flag;

    /////////

    hypre_base(std::shared_ptr<part::part> _part)
        : part(_part), row_indx(0), mg_precond_flag(false) {
        hA = nullptr;
        hb = hx = nullptr;

        parcsr_A = nullptr;
        par_b = par_x = nullptr;

        solver = precond = nullptr;
    }

    /////////

    void set_matrix_diag_block(const matrix::matrix_block &diag, const uint32_t &offset,
                               const uint32_t &size);
    void set_matrix_offd_block(const matrix::matrix_block &offd,
                               const std::pair<uint64_t, uint64_t> &range);

    void assemble_matrix(const matrix::matrix &m);
    void assemble_vector(HYPRE_IJVector &hb, const vector::vector &b);

    void get_vector_data(HYPRE_IJVector &hb, vector::vector &b);

    void get_matrix_object() { HYPRE_IJMatrixGetObject(hA, (void **)&parcsr_A); }

    void get_vector_objects() {
        HYPRE_IJVectorGetObject(hb, (void **)&par_b);
        HYPRE_IJVectorGetObject(hx, (void **)&par_x);
    }

    void get_objects() {
        get_matrix_object();
        get_vector_objects();
    }

    void destroy_matrix() { HYPRE_IJMatrixDestroy(hA); }

    void destroy_vectors() {
        HYPRE_IJVectorDestroy(hx);
        HYPRE_IJVectorDestroy(hb);
    }

    void destroy_objects() {
        destroy_matrix();
        destroy_vectors();
    }

    void create_bicgstab_solver(HYPRE_Solver &sol, const params::global_param_list &global_list) {
        std::string solver_type;
        const auto &list = global_list.get_if<std::string>(
            {{"method", "BiCGStab"}, {"method", "PBiCGStab"}}, solver_type);

        HYPRE_ParCSRBiCGSTABCreate(*((MPI_Comm *)id.get_comm()), &sol);

        //  Set some parameters (See Reference Manual for more parameters)
        HYPRE_BiCGSTABSetMaxIter(sol, list.get_value<uint16_t>("max_iters"));
        HYPRE_BiCGSTABSetTol(sol, list.get_value<float32_t>("rel_tolerance"));
        HYPRE_BiCGSTABSetAbsoluteTol(sol, list.get_value<float32_t>("abs_tolerance"));

        if (list.get_value<uint16_t>("convergence_info"))
            HYPRE_BiCGSTABSetPrintLevel(sol, 2); // prints out the iteration info
        else
            HYPRE_BiCGSTABSetPrintLevel(sol, 0);
        HYPRE_BiCGSTABSetLogging(sol, 1); // needed to get run info later
    }

    void create_multigrid_solver(HYPRE_Solver &sol, const params::global_param_list &global_list) {
        std::string solver_type;
        const auto &list = global_list.get_if<std::string>({{"method", "MultiGrid"}}, solver_type);

        HYPRE_BoomerAMGCreate(&sol);

        HYPRE_BoomerAMGSetMaxIter(sol, list.get_value<uint16_t>("max_iters"));
        HYPRE_BoomerAMGSetTol(sol, list.get_value<float32_t>("rel_tolerance"));

        uint16_t hypre_log = list.get_value<uint16_t>("hypre_log");
        if (hypre_log) {
            HYPRE_BoomerAMGSetPrintLevel(sol, 3);
            HYPRE_BoomerAMGSetLogging(sol, 1);
        } else if (list.get_value<uint16_t>("convergence_info")) {
            HYPRE_BoomerAMGSetPrintLevel(sol, 2);
            HYPRE_BoomerAMGSetLogging(sol, 1);
        } else {
            HYPRE_BoomerAMGSetPrintLevel(sol, 0);
            HYPRE_BoomerAMGSetLogging(sol, 0);
        }

        HYPRE_BoomerAMGSetConvergeType(sol, 1); // res = ||r_n|| / ||r_0||

        // HYPRE_BoomerAMGSetRestriction(sol, 0);  // ???
        // HYPRE_BoomerAMGSetRAP2(solver, 1);
        // Fixing multigrid hierarchy for debuging purposes:
        // HYPRE_BoomerAMGSetCoarsenType(solver, 9);

        auto max_levels = list.get_value<uint16_t>("mg_max_levels");
        HYPRE_BoomerAMGSetMaxLevels(sol, max_levels);
        HYPRE_BoomerAMGSetMaxCoarseSize(sol, list.get_value<uint16_t>("mg_coarse_matrix_size"));

        HYPRE_BoomerAMGSetStrongThreshold(sol, list.get_value<float32_t>("mg_strength_threshold"));
        HYPRE_BoomerAMGSetTruncFactor(sol, list.get_value<float32_t>("mg_trunc_factor"));

        HYPRE_BoomerAMGSetNumPaths(sol, list.get_value<uint16_t>("mg_num_paths"));

        HYPRE_BoomerAMGSetCoarsenType(sol, list.get_value<uint16_t>("mg_coarsening_type"));
        HYPRE_BoomerAMGSetInterpType(sol, list.get_value<uint16_t>("mg_interpolation_type"));

        HYPRE_BoomerAMGSetMaxRowSum(sol, list.get_value<float32_t>("mg_max_row_sum"));
        HYPRE_BoomerAMGSetPMaxElmts(sol, list.get_value<uint16_t>("mg_Pmax_elements"));
        HYPRE_BoomerAMGSetNonGalerkinTol(sol, list.get_value<float32_t>("mg_nonGalerkin_tol"));

        HYPRE_BoomerAMGSetAggNumLevels(sol, list.get_value<uint16_t>("mg_agg_num_levels"));
        HYPRE_BoomerAMGSetAggInterpType(sol, list.get_value<uint16_t>("mg_agg_interpolation_type"));

        HYPRE_BoomerAMGSetAggTruncFactor(sol, list.get_value<float32_t>("mg_agg_trunc_factor"));
        HYPRE_BoomerAMGSetAggP12TruncFactor(sol,
                                            list.get_value<float32_t>("mg_agg_P12_trunc_factor"));

        HYPRE_BoomerAMGSetAggPMaxElmts(sol, list.get_value<uint16_t>("mg_agg_Pmax_elements"));
        HYPRE_BoomerAMGSetAggP12MaxElmts(sol, list.get_value<uint16_t>("mg_agg_P12max_elements"));

        /////////

        HYPRE_BoomerAMGSetRelaxOrder(sol, 0);
        HYPRE_BoomerAMGSetNumSweeps(sol, 1);
        //        HYPRE_BoomerAMGSetOldDefault(sol);

        for (int level = 0; level < max_levels; level++) {
            auto per_level_list = global_list.get(solver_type, level);
            auto mg_nonGalerkin_tol = per_level_list.get_value<float32_t>("mg_nonGalerkin_tol");
            HYPRE_BoomerAMGSetLevelNonGalerkinTol(sol, mg_nonGalerkin_tol, level);
        }

        /////////

        std::map<std::string, uint16_t> solver_types{
            {"pre_smoother", 1}, {"post_smoother", 2}, {"coarse_grid_solver", 3}};
        for (const auto &i : solver_types) {
            std::string solver_type = i.first;
            uint16_t solver_type_idx = i.second;
            auto &solver_list = global_list.get(solver_type);
            std::string method = solver_list.get_value<std::string>("method");
            if (method == "Jacobi") {
                HYPRE_BoomerAMGSetCycleRelaxType(sol, 0, solver_type_idx);
                HYPRE_BoomerAMGSetRelaxWt(sol, solver_list.get_value<float32_t>("relax_factor"));
                HYPRE_BoomerAMGSetCycleNumSweeps(sol, solver_list.get_value<uint16_t>("max_iters"),
                                                 solver_type_idx);
            } else if (method == "HSGS") {
                HYPRE_BoomerAMGSetCycleRelaxType(sol, 6, solver_type_idx);
                HYPRE_BoomerAMGSetRelaxWt(sol, solver_list.get_value<float32_t>("relax_factor"));
                HYPRE_BoomerAMGSetCycleNumSweeps(sol, solver_list.get_value<uint16_t>("max_iters"),
                                                 solver_type_idx);
            } else if (method == "Chebyshev") {
                HYPRE_BoomerAMGSetCycleRelaxType(sol, 16, solver_type_idx);
                HYPRE_BoomerAMGSetChebyOrder(sol,
                                             solver_list.get_value<uint16_t>("polynomial_order"));
            } else if ((method == "Direct") && (solver_type_idx == 3)) { // coarse grid solver
                HYPRE_BoomerAMGSetCycleRelaxType(sol, 9, solver_type_idx);
            } else {
                assert(0);
            }
        }
    }

    void setup_bicgstab_solver() {
        double t1, t2;
        mpi::barrier(mpi::GLOBAL);
        t1 = sys::timer();

        if (mg_precond_flag)
            HYPRE_BiCGSTABSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                                     (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);

        HYPRE_ParCSRBiCGSTABSetup(solver, parcsr_A, par_b, par_x);

        mpi::barrier(mpi::GLOBAL);
        t2 = sys::timer();
    }

    void setup_multigrid_solver() {
        double t1, t2;
        mpi::barrier(mpi::GLOBAL);
        t1 = sys::timer();

        if (mg_precond_flag)
            HYPRE_BoomerAMGSetup(precond, parcsr_A, NULL, NULL);
        else
            HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);

        mpi::barrier(mpi::GLOBAL);
        t2 = sys::timer();
    }

    void bicgstab_solver() {
        double t2, t3;
        mpi::barrier(mpi::GLOBAL);
        t2 = sys::timer();

        HYPRE_ParCSRBiCGSTABSolve(solver, parcsr_A, par_b, par_x);

        mpi::barrier(mpi::GLOBAL);
        t3 = sys::timer();

        int num_iterations;
        double final_res_norm;
        HYPRE_BiCGSTABGetNumIterations(solver, &num_iterations);
        HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

        XAMG::out << "Hypre solver time = " << t3 - t2 << " sec" << std::endl;
    }

    void multigrid_solver() {
        double t2, t3;
        mpi::barrier(mpi::GLOBAL);
        t2 = sys::timer();

        HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);

        mpi::barrier(mpi::GLOBAL);
        t3 = sys::timer();

        int num_iterations;
        double final_res_norm;
        HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
        HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

        XAMG::out << "Hypre solver time = " << t3 - t2 << " sec" << std::endl;
    }

    void destroy_bicgstab_solver() {
        if (mg_precond_flag)
            HYPRE_BoomerAMGDestroy(precond);
        HYPRE_ParCSRBiCGSTABDestroy(solver);
    }

    void destroy_multigrid_solver() { HYPRE_BoomerAMGDestroy(solver); }

    /////////

    template <typename F>
    void parse_matrix(matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t> &mat_csr,
                      const hypre_ParCSRMatrix *hypre_matrix);

    void get_part(part::part &part, const hypre_ParCSRMatrix *hypre_matrix);

    void create_hierarchy(const params::global_param_list &global_list);
    template <typename F>
    void parse_hierarchy(std::vector<matrix::mg_layer> &mg_tree, const bool reduced_prec);
};

} // namespace hypre
} // namespace XAMG

#include "detail/hypre_base.inl"
