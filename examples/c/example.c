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

#include "xamg_c_api.h"

#include <unistd.h>

#ifndef XAMG_NV
#define XAMG_NV 2
#endif

const uint16_t NV = XAMG_NV;

void generate_system(XAMG_csr_matrix_id m_csr_id, XAMG_vector_id x0_id, XAMG_vector_id b0_id,
                     int nx, int ny, int nz, int nv) {
    int proc = XAMG_id_get_proc();
    int nprocs = XAMG_id_get_nprocs();

    int nrows = nx * ny * nz;
    int block_size = nrows / nprocs;
    int block_offset = block_size * proc;
    if (proc == nprocs - 1)
        block_size = nrows - block_offset;
    int block_nonzeros = 7 * block_size;

    int *row = (int *)calloc(block_size + 1, sizeof(int));
    int *col = (int *)calloc(block_nonzeros, sizeof(int));
    double *val = (double *)calloc(block_nonzeros, sizeof(double));

    /////////

    double hx = 1.0 / nx;
    double hy = 1.0 / ny;
    double hz = 1.0 / nz;

    double q1 = 1.0 / hx / hx;
    double q2 = 1.0 / hy / hy;
    double q3 = 1.0 / hz / hz;

    int i;
    int cntr = 0;
    row[0] = cntr;
    for (i = 0; i < block_size; ++i) {
        int indx = block_offset + i;
        int ii = indx % nx;
        int jj = (indx / nx) % ny;
        int kk = indx / (nx * ny);
        int row_start = cntr;
        col[cntr] = indx;
        val[cntr] = 0;
        ++cntr;

        if (kk > 0) {
            val[row_start] += q3;
            col[cntr] = indx - nx * ny;
            val[cntr] = -q3;
            ++cntr;
        } else if (nz > 1)
            val[row_start] += 2.0 * q3;

        if (jj > 0) {
            val[row_start] += q2;
            col[cntr] = indx - nx;
            val[cntr] = -q2;
            ++cntr;
        } else if (ny > 1)
            val[row_start] += 2.0 * q2;

        if (ii > 0) {
            val[row_start] += q1;
            col[cntr] = indx - 1;
            val[cntr] = -q1;
            ++cntr;
        } else if (nx > 1)
            val[row_start] += 2.0 * q1;

        if (ii < nx - 1) {
            val[row_start] += q1;
            col[cntr] = indx + 1;
            val[cntr] = -q1;
            ++cntr;
        } else if (nx > 1)
            val[row_start] += 2.0 * q1;

        if (jj < ny - 1) {
            val[row_start] += q2;
            col[cntr] = indx + nx;
            val[cntr] = -q2;
            ++cntr;
        } else if (ny > 1)
            val[row_start] += 2.0 * q2;

        if (kk < nz - 1) {
            val[row_start] += q3;
            col[cntr] = indx + nx * ny;
            val[cntr] = -q3;
            ++cntr;
        } else if (nz > 1)
            val[row_start] += 2.0 * q3;

        row[i + 1] = cntr;
    }

    XAMG_csr_matrix_alloc(block_size, cntr, m_csr_id);
    XAMG_csr_matrix_fill(row, col, val, m_csr_id);
    XAMG_csr_matrix_offset(block_offset, 0, m_csr_id);

    //////////

    XAMG_vector_alloc_d(block_size, nv, x0_id);
    XAMG_vector_alloc_d(block_size, nv, b0_id);

    XAMG_vector_offset(block_offset, x0_id);
    XAMG_vector_offset(block_offset, b0_id);

    XAMG_vector_set_val_d(0.0, x0_id);
    XAMG_vector_set_val_d(1.0, b0_id);
}

int main(int argc, char *argv[]) {

    XAMG_init(argc, argv, "");

    int xamg_nv = XAMG_get_nv();
    if (xamg_nv != NV) {
        printf("Number of rhs vectors required by the test code (%d) does not match the one in "
               "XAMG API (%d)\n",
               NV, xamg_nv);
        return EXIT_FAILURE;
    }

    XAMG_vector_id x0_id, b0_id;
    // generate_system function operates with local vectors:
    XAMG_vector_create(&x0_id, XAMG_MEM_LOCAL); // local
    XAMG_vector_create(&b0_id, XAMG_MEM_LOCAL); // local

    XAMG_csr_matrix_id m_csr_id;
    XAMG_csr_matrix_create(&m_csr_id);

    generate_system(m_csr_id, x0_id, b0_id, 10, 10, 10, NV);

    XAMG_matrix_id m_id;
    XAMG_matrix_create(&m_id);

    XAMG_part_id p_id;
    XAMG_part_create(&p_id);
    XAMG_part_construct(XAMG_csr_matrix_get_nrows(m_csr_id), p_id);

    XAMG_matrix_set_part(p_id, m_id);
    XAMG_matrix_construct(m_csr_id, m_id);

    XAMG_csr_matrix_destroy(m_csr_id);

    XAMG_param_id par_id;
    XAMG_param_create(&par_id);
    XAMG_param_add_value_s("solver", "method", "PBiCGStab", par_id);
    XAMG_param_add_value_i("solver", "convergence_info", 1, par_id);
    XAMG_param_add_value_d("solver", "rel_tolerance", 1.e-6, par_id);
    XAMG_param_add_value_d("solver", "abs_tolerance", 1.e-6, par_id);
    XAMG_param_add_value_s("preconditioner", "method", "MultiGrid", par_id);
    XAMG_param_add_value_i("preconditioner", "hypre_per_level_hierarchy", 1, par_id);

    XAMG_param_override_id ov_id;
    XAMG_param_override_create(&ov_id);
    XAMG_param_override_add_value("mg_coarsening_type", "8@lev0-1", ov_id);
    XAMG_param_override_add_value("mg_interpolation_type", "6@lev0-1", ov_id);
    XAMG_param_override_apply("preconditioner", par_id, ov_id);
    XAMG_param_override_destroy(ov_id);

    XAMG_param_set_default(par_id);

    ////

    XAMG_vector_id x_id, b_id;
    XAMG_vector_create(&x_id, XAMG_MEM_DISTRIBUTED); // distributed
    XAMG_vector_create(&b_id, XAMG_MEM_DISTRIBUTED); // distributed
    XAMG_vector_set_part(p_id, x_id);
    XAMG_vector_set_part(p_id, b_id);

    int vec_size = XAMG_part_get_numa_size(p_id);
    XAMG_vector_alloc_d(vec_size, NV, x_id);
    XAMG_vector_alloc_d(vec_size, NV, b_id);
    XAMG_vector_upload_v(x0_id, x_id);
    XAMG_vector_upload_v(b0_id, b_id);

    XAMG_solver_id sol_id;
    XAMG_solver_create2(m_id, par_id, x_id, b_id, &sol_id);

    XAMG_barrier();
    double t1 = XAMG_timer();

    XAMG_param_print(par_id);

    XAMG_solver_solve1(NV, sol_id);

    XAMG_barrier();
    double t2 = XAMG_timer();

    printf("Solver time: %.6f\n", t2 - t1);

    /////////

    XAMG_param_change_value_d("solver", "rel_tolerance", 1.e-8, par_id);
    XAMG_param_change_value_d("solver", "abs_tolerance", 1.e-8, par_id);
    XAMG_solver_renew_params(sol_id, par_id);

    XAMG_vector_upload_v(x0_id, x_id);

    XAMG_barrier();
    t1 = XAMG_timer();

    XAMG_param_print(par_id);

    XAMG_solver_solve1(NV, sol_id);

    XAMG_barrier();
    t2 = XAMG_timer();

    printf("Solver time: %.6f\n", t2 - t1);

    /////////

    XAMG_vector_destroy(x_id);
    XAMG_vector_destroy(b_id);

    XAMG_matrix_destroy(m_id);

    XAMG_solver_destroy(sol_id);

    XAMG_finalize();
}
