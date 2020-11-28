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

#ifndef XAMG_C_API_H_
#define XAMG_C_API_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef size_t XAMG_param_id;
typedef size_t XAMG_param_override_id;
typedef size_t XAMG_part_id;
typedef size_t XAMG_vector_id;
typedef size_t XAMG_csr_matrix_id;
typedef size_t XAMG_matrix_id;
typedef size_t XAMG_solver_id;

typedef enum { XAMG_MEM_LOCAL, XAMG_MEM_SHARED, XAMG_MEM_DISTRIBUTED } vector_alloc_mode;

int XAMG_init(int argc, char **argv, char *conf);
int XAMG_finalize();

int XAMG_get_nv();

int XAMG_id_get_proc();
int XAMG_id_get_nprocs();
void *XAMG_id_get_comm_ptr();

#define XAMG_id_get_comm() (*((MPI_Comm *)XAMG_id_get_comm_ptr()))

int XAMG_param_create(XAMG_param_id *);
int XAMG_param_add_value_s(char *solver_type, char *key, char *value, XAMG_param_id pid);
int XAMG_param_add_value_i(char *solver_type, char *key, uint16_t value, XAMG_param_id pid);
int XAMG_param_add_value_d(char *solver_type, char *key, float value, XAMG_param_id pid);

int XAMG_param_forced_change_value_s(char *solver_type, char *key, char *value, XAMG_param_id pid);
int XAMG_param_forced_change_value_i(char *solver_type, char *key, uint16_t value,
                                     XAMG_param_id pid);
int XAMG_param_forced_change_value_d(char *solver_type, char *key, float value, XAMG_param_id pid);

int XAMG_param_change_value_s(char *solver_type, char *key, char *value, XAMG_param_id pid);
int XAMG_param_change_value_i(char *solver_type, char *key, uint16_t value, XAMG_param_id pid);
int XAMG_param_change_value_d(char *solver_type, char *key, float value, XAMG_param_id pid);

int XAMG_param_override_create(XAMG_param_override_id *);
int XAMG_param_override_add_value(char *key, char *val, XAMG_param_override_id ovid);
int XAMG_param_override_apply(char *solver_type, XAMG_param_id pid, XAMG_param_override_id ovid);
int XAMG_param_override_destroy(XAMG_param_override_id);

void XAMG_param_set_default(XAMG_param_id);
int XAMG_param_print(XAMG_param_id);
int XAMG_param_destroy(XAMG_param_id);

int XAMG_part_create(XAMG_part_id *);
int XAMG_part_construct(int, XAMG_part_id);
int XAMG_part_get_numa_size(XAMG_part_id);
int XAMG_part_destroy(XAMG_part_id);

int XAMG_vector_create(XAMG_vector_id *, vector_alloc_mode);
int XAMG_vector_offset(uint64_t, XAMG_vector_id);
int XAMG_vector_alloc_d(int, uint16_t, XAMG_vector_id);
// int XAMG_vector_alloc_i(int, int, uint16_t, XAMG_vector_id);
int XAMG_vector_set_part(XAMG_part_id, XAMG_vector_id);
int XAMG_vector_set_val_d(double, XAMG_vector_id);
int XAMG_vector_download_d(double *, uint64_t, uint64_t, uint16_t, XAMG_vector_id);
//  To be replaced with XAMG_vector_copy
int XAMG_vector_upload_v(XAMG_vector_id, XAMG_vector_id);
int XAMG_vector_upload_d(double *, uint64_t, uint64_t, uint16_t, XAMG_vector_id);
// int XAMG_vector_copy_v(XAMG_vector_id, XAMG_vector_id);
int XAMG_vector_destroy(XAMG_vector_id);

int XAMG_csr_matrix_create(XAMG_csr_matrix_id *);
int XAMG_csr_matrix_alloc(uint64_t, uint64_t, XAMG_csr_matrix_id);
int XAMG_csr_matrix_fill(int *, int *, double *, XAMG_matrix_id);
int XAMG_csr_matrix_offset(int, int, XAMG_matrix_id);
uint32_t XAMG_csr_matrix_get_nrows(XAMG_csr_matrix_id);
int XAMG_csr_matrix_destroy(XAMG_csr_matrix_id);

int XAMG_matrix_create(XAMG_matrix_id *);
int XAMG_matrix_set_part(XAMG_part_id, XAMG_matrix_id);
int XAMG_matrix_construct(XAMG_csr_matrix_id mcsrid, XAMG_matrix_id mid);
int XAMG_matrix_destroy(XAMG_matrix_id);

int XAMG_solver_create1(XAMG_matrix_id, XAMG_param_id, XAMG_solver_id *);
int XAMG_solver_create2(XAMG_matrix_id, XAMG_param_id, XAMG_vector_id, XAMG_vector_id,
                        XAMG_solver_id *);
int XAMG_solver_renew_params(XAMG_param_id, XAMG_solver_id);
int XAMG_solver_solve1(uint16_t, XAMG_solver_id);
int XAMG_solver_solve2(uint16_t, XAMG_vector_id, XAMG_vector_id, XAMG_solver_id);
int XAMG_solver_get_convergence_info(int *, uint16_t, XAMG_solver_id);
int XAMG_solver_destroy(XAMG_solver_id);

double XAMG_timer();
int XAMG_barrier();

#endif /* XAMG_C_API_H_ */
