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

#include <string>
#include "xamg/xamg_headers.h"
#include "xamg/xamg_types.h"

#include "xamg/init.h"

#include "xamg/primitives/vector/vector.h"
#include "xamg/primitives/matrix/matrix.h"

// TODO: hide these includes
#include "xamg/blas/blas.h"
#include "xamg/blas2/blas2.h"

#include "xamg/solvers/solver.h"

#include "xamg/param/params.h"
#include "xamg/io/logout.h"

#include <mpi.h>

typedef size_t XAMG_param_id;
typedef size_t XAMG_param_override_id;
typedef size_t XAMG_part_id;
typedef size_t XAMG_vector_id;
typedef size_t XAMG_matrix_id;
typedef size_t XAMG_csr_matrix_id;
typedef size_t XAMG_solver_id;

typedef enum { XAMG_MEM_LOCAL, XAMG_MEM_SHARED, XAMG_MEM_DISTRIBUTED } vector_alloc_mode;

#ifndef XAMG_NV
const uint16_t CONST_NV = 1;
#else
const uint16_t CONST_NV = XAMG_NV;
#endif

//------------------------------------

extern ID id;

using param_t = XAMG::params::global_param_list;
std::vector<param_t *> param_list;

using param_override_t = std::map<std::string, std::string>;
std::vector<param_override_t *> ovlist;

using part_t = std::shared_ptr<XAMG::part::part>;
std::vector<part_t> part_list;

using vector_t = XAMG::vector::vector;
std::vector<vector_t *> vlist;

using csr_matrix_t = XAMG::matrix::csr_matrix<float64_t, uint32_t, uint32_t, uint32_t, uint32_t>;
std::vector<csr_matrix_t *> mcsrlist;

using matrix_t = XAMG::matrix::matrix;
std::vector<matrix_t *> mlist;

using solver_t = std::shared_ptr<XAMG::solver::base_solver<float64_t, CONST_NV>>;
std::vector<solver_t> slist;

namespace helper {

void check_rhs(uint16_t nv) {
    //    XAMG::out << CONST_NV << " " << nv << std::endl;
    assert(CONST_NV == nv);
}

template <typename T1, typename T2>
void vector_fill(XAMG::vector::vector &vec, T2 *from, size_t size) {
    assert(vec.sharing_mode == XAMG::mem::CORE);
    vec.check(XAMG::vector::vector::allocated);

    auto vec_ptr = vec.get_aligned_ptr<T1>();
    for (uint64_t i = 0; i < vec.size * vec.nv; ++i)
        vec_ptr[i] = (T1)from[i];

    vec.if_initialized = true;
}

template <typename T>
int vector_set_val(T val, uint16_t nv, XAMG_vector_id vid) {
    check_rhs(nv);
    vector_t *v = vlist[vid];

    XAMG::blas::set_const<T, CONST_NV>(*v, val, true);
    return 0;
}

template <typename T>
int xamg_param_add_value(char *solver_type, char *key, T value, XAMG_param_id pid) {
    param_t *p = param_list[pid];

    if (!p->find(solver_type)) {
        XAMG::params::param_list new_list;
        new_list.add_value<T>(key, value);
        p->add(solver_type, new_list);
    } else {
        p->get(solver_type).add_value<T>(key, value);
    }

    return 0;
}

template <typename T>
int xamg_param_change_value(char *solver_type, char *key, T value, XAMG_param_id pid) {
    param_t *p = param_list[pid];

    p->change_value<T>(solver_type, key, value);

    return 0;
}

template <typename T>
int xamg_param_forced_change_value(char *solver_type, char *key, T value, XAMG_param_id pid) {
    param_t *p = param_list[pid];

    p->forced_change_value<T>(solver_type, key, value);

    return 0;
}
} // namespace helper

//!!! C-source
extern "C" {

int XAMG_id_get_proc() {
    return id.gl_proc;
}

int XAMG_id_get_nprocs() {
    return id.gl_nprocs;
}

// MPI_Comm XAMG_id_get_comm() {
//    return *((MPI_Comm*)id.get_comm());
//}
void *XAMG_id_get_comm_ptr() {
    return id.get_comm();
}

/////////
//  Param list object:

int XAMG_param_create(XAMG_param_id *pid) {
    param_t *p = new param_t();

    param_list.push_back(p);
    *pid = param_list.size() - 1;
    return 0;
}

int XAMG_param_add_value_s(char *solver_type, char *key, char *value, XAMG_param_id pid) {
    helper::xamg_param_add_value<std::string>(solver_type, key, value, pid);
    return 0;
}

int XAMG_param_add_value_i(char *solver_type, char *key, uint16_t value, XAMG_param_id pid) {
    helper::xamg_param_add_value<uint16_t>(solver_type, key, value, pid);
    return 0;
}

int XAMG_param_add_value_d(char *solver_type, char *key, float value, XAMG_param_id pid) {
    helper::xamg_param_add_value<float32_t>(solver_type, key, value, pid);
    return 0;
}

int XAMG_param_change_value_s(char *solver_type, char *key, char *value, XAMG_param_id pid) {
    helper::xamg_param_change_value<std::string>(solver_type, key, value, pid);
    return 0;
}

int XAMG_param_change_value_i(char *solver_type, char *key, uint16_t value, XAMG_param_id pid) {
    helper::xamg_param_change_value<uint16_t>(solver_type, key, value, pid);
    return 0;
}

int XAMG_param_change_value_d(char *solver_type, char *key, float value, XAMG_param_id pid) {
    helper::xamg_param_change_value<float32_t>(solver_type, key, value, pid);
    return 0;
}

int XAMG_param_forced_change_value_s(char *solver_type, char *key, char *value, XAMG_param_id pid) {
    helper::xamg_param_forced_change_value<std::string>(solver_type, key, value, pid);
    return 0;
}

int XAMG_param_forced_change_value_i(char *solver_type, char *key, uint16_t value,
                                     XAMG_param_id pid) {
    helper::xamg_param_forced_change_value<uint16_t>(solver_type, key, value, pid);
    return 0;
}

int XAMG_param_forced_change_value_d(char *solver_type, char *key, float value, XAMG_param_id pid) {
    helper::xamg_param_forced_change_value<float32_t>(solver_type, key, value, pid);
    return 0;
}

void XAMG_param_set_default(XAMG_param_id pid) {
    auto &par = param_list[pid];
    par->set_defaults();
}

int XAMG_param_print(XAMG_param_id pid) {
    auto &par = param_list[pid];
    par->print();
    return 0;
}

int XAMG_param_destroy(XAMG_param_id pid) {
    if (param_list[pid] != nullptr) {
        delete param_list[pid];
        param_list[pid] = nullptr;
    }
    return 0;
}

int XAMG_param_override_create(XAMG_param_override_id *ovid) {
    param_override_t *ov = new param_override_t;
    ovlist.push_back(ov);
    *ovid = ovlist.size() - 1;
    return 0;
}

int XAMG_param_override_add_value(char *key, char *val, XAMG_param_override_id ovid) {
    auto &ov = ovlist[ovid];
    ov->insert({key, val});
    return 0;
}

int XAMG_param_override_apply(char *solver_type, XAMG_param_id pid, XAMG_param_override_id ovid) {
    auto &p = param_list[pid];
    auto &ov = ovlist[ovid];
    p->add_override(solver_type, *ov);
    return 0;
}

int XAMG_param_override_destroy(XAMG_param_override_id ovid) {
    if (ovlist[ovid] != nullptr) {
        delete ovlist[ovid];
        ovlist[ovid] = nullptr;
    }
    return 0;
}

/////////
//  Part object

int XAMG_part_create(XAMG_part_id *pid) {
    part_t p = XAMG::part::get_shared_part();
    part_list.push_back(p);
    *pid = part_list.size() - 1;
    return 0;
}

int XAMG_part_construct(int size, XAMG_part_id pid) {
    auto &p = part_list[pid];
    p->get_part(size);
    return 0;
}

int XAMG_part_get_numa_size(XAMG_part_id pid) {
    auto &p = part_list[pid];
    return (p->numa_layer.block_size[id.nd_numa]);
}

int XAMG_part_destroy(XAMG_part_id pid) {
    if (part_list[pid] != nullptr) {
        //        delete part_list[pid];
        part_list[pid] = nullptr;
    }
    return 0;
}

/////////
//  Vector object:

int XAMG_vector_create(XAMG_vector_id *vid, vector_alloc_mode alloc_mode) {
    vector_t *v = new vector_t((XAMG::mem::allocation)alloc_mode);

    vlist.push_back(v);
    *vid = vlist.size() - 1;
    return 0;
}

int XAMG_vector_set_part(XAMG_part_id pid, XAMG_vector_id vid) {
    auto &p = part_list[pid];
    auto &v = vlist[vid];
    v->set_part(p);
    return 0;
}

int XAMG_vector_offset(uint64_t offset, XAMG_vector_id vid) {
    auto &v = vlist[vid];
    v->ext_offset = offset;
    return 0;
}

int XAMG_vector_alloc_d(int size, uint16_t nv, XAMG_vector_id vid) {
    helper::check_rhs(nv);
    auto &v = vlist[vid];

    v->alloc<double>(size, nv);
    return 0;
}

// int XAMG_vector_alloc_i(int size, int offset, uint16_t nv, XAMG_vector_id vid) {
//    helper::check_rhs(nv);
//    auto &v = vlist[vid];
//    assert(v->sharing_mode == XAMG::mem::NUMA_NODE);
//
//    v->alloc<uint32_t>(size, nv);
//    v->ext_offset = offset;
//    return 0;
//}

int XAMG_vector_set_val_d(double val, XAMG_vector_id vid) {
    auto &v = vlist[vid];

    //    assume value is the same for the whole vector!
    XAMG::blas::set_const<double, CONST_NV>(*v, val, true);
    return 0;
}

int XAMG_vector_upload_d(double *ptr, uint64_t block_size, uint64_t block_offset, uint16_t nv,
                         XAMG_vector_id vid) {
    helper::check_rhs(nv);
    auto &vec = vlist[vid];

    XAMG::mpi::barrier(XAMG::mpi::INTRA_NUMA);
    XAMG::blas::upload<float64_t, CONST_NV>(*vec, (float64_t *)ptr, block_size, block_offset);
    XAMG::mpi::barrier(XAMG::mpi::INTRA_NUMA);
    return 0;
}

int XAMG_vector_download_d(double *ptr, uint64_t block_size, uint64_t block_offset, uint16_t nv,
                           XAMG_vector_id vid) {
    helper::check_rhs(nv);
    auto &vec = vlist[vid];
    vec->check(XAMG::vector::vector::allocated);

    auto vec_ptr = vec->get_aligned_ptr<float64_t>();
    XAMG::mpi::barrier(XAMG::mpi::INTRA_NUMA);
    memcpy(ptr, vec_ptr + (block_offset - vec->global_numa_offset()) * nv,
           block_size * nv * sizeof(float64_t));
    XAMG::mpi::barrier(XAMG::mpi::INTRA_NUMA);
    return 0;
}

int XAMG_vector_upload_v(XAMG_vector_id vid_in, XAMG_vector_id vid_out) {
    auto &v_in = vlist[vid_in];
    auto &v_out = vlist[vid_out];

    assert(v_in->sharing_mode == XAMG::mem::CORE);
    assert(v_out->sharing_mode == XAMG::mem::NUMA_NODE);

    XAMG::blas::copy<double, CONST_NV>(*v_in, *v_out);
    return 0;
}

int XAMG_vector_destroy(XAMG_vector_id vid) {
    if (vlist[vid] != nullptr) {
        delete vlist[vid];
        vlist[vid] = nullptr;
    }
    return 0;
}

/////////
//  CSR matrix object:

int XAMG_csr_matrix_create(XAMG_csr_matrix_id *mcsrid) {
    csr_matrix_t *mcsr = new csr_matrix_t();

    mcsrlist.push_back(mcsr);
    *mcsrid = mcsrlist.size() - 1;
    return 0;
}

int XAMG_csr_matrix_alloc(uint64_t nrows, uint64_t nnz, XAMG_matrix_id mcsrid) {
    auto &mcsr = mcsrlist[mcsrid];

    mcsr->nrows = nrows;
    mcsr->nonzeros = nnz;
    mcsr->alloc();
    return 0;
}

int XAMG_csr_matrix_fill(int *row, int *col, double *val, XAMG_matrix_id mcsrid) {
    auto &mcsr = mcsrlist[mcsrid];
    assert(mcsr->sharing_mode == XAMG::mem::CORE);
    assert(mcsr->if_indexed == false);

    helper::vector_fill<uint32_t, int>(mcsr->row, row, mcsr->nrows + 1);
    helper::vector_fill<uint32_t, int>(mcsr->col, col, mcsr->nonzeros);
    helper::vector_fill<float64_t, double>(mcsr->val, val, mcsr->nonzeros);
    return 0;
}

int XAMG_csr_matrix_offset(int row_offset, int col_offset, XAMG_matrix_id mcsrid) {
    auto &mcsr = mcsrlist[mcsrid];

    mcsr->block_row_offset = row_offset;
    mcsr->block_col_offset = col_offset;
    return 0;
}

uint32_t XAMG_csr_matrix_get_nrows(XAMG_csr_matrix_id mcsrid) {
    auto &mcsr = mcsrlist[mcsrid];
    return mcsr->nrows;
}

int XAMG_csr_matrix_destroy(XAMG_csr_matrix_id mcsrid) {
    if (mcsrlist[mcsrid] != nullptr) {
        delete mcsrlist[mcsrid];
        mcsrlist[mcsrid] = nullptr;
    }
    return 0;
}

/////////
//  Matrix object:

int XAMG_matrix_create(XAMG_matrix_id *mid) {
    matrix_t *m = new matrix_t(XAMG::mem::DISTRIBUTED);

    mlist.push_back(m);
    *mid = mlist.size() - 1;
    return 0;
}

int XAMG_matrix_set_part(XAMG_part_id pid, XAMG_matrix_id mid) {
    auto &p = part_list[pid];
    auto &m = mlist[mid];

    m->set_part(p);
    return 0;
}

int XAMG_matrix_construct(XAMG_csr_matrix_id mcsrid, XAMG_matrix_id mid) {
    auto &m = mlist[mid];
    auto &mcsr = mcsrlist[mcsrid];

    m->construct(*mcsr);
    return 0;
}

int XAMG_matrix_destroy(XAMG_matrix_id mid) {
    if (mlist[mid] != nullptr) {
        delete mlist[mid];
        mlist[mid] = nullptr;
    }
    return 0;
}

/////////
//  Solver object:

int XAMG_solver_create1(XAMG_matrix_id mid, XAMG_param_id pid, XAMG_solver_id *sid) {
    auto &m = mlist[mid];
    auto &par = param_list[pid];

    slist.push_back(XAMG::solver::construct_solver_hierarchy<float64_t, CONST_NV>(*par, *m));
    *sid = slist.size() - 1;
    return 0;
}

int XAMG_solver_create2(XAMG_matrix_id mid, XAMG_param_id pid, XAMG_vector_id xid,
                        XAMG_vector_id bid, XAMG_solver_id *sid) {
    auto &m = mlist[mid];
    auto &par = param_list[pid];
    auto &x = vlist[xid];
    auto &b = vlist[bid];

    slist.push_back(
        XAMG::solver::construct_solver_hierarchy<float64_t, CONST_NV>(*par, *m, *x, *b));
    *sid = slist.size() - 1;
    return 0;
}

int XAMG_solver_renew_params(XAMG_param_id pid, XAMG_solver_id sid) {
    auto &sol = slist[sid];
    auto &par = param_list[pid];

    sol->renew_params(*par);
    return 0;
}

int XAMG_solver_solve1(uint16_t nv, XAMG_solver_id sid) {
    helper::check_rhs(nv);
    auto &sol = slist[sid];

    sol->solve();
    return 0;
}

int XAMG_solver_solve2(uint16_t nv, XAMG_vector_id xid, XAMG_vector_id bid, XAMG_solver_id sid) {
    helper::check_rhs(nv);
    auto &sol = slist[sid];
    auto &x = vlist[xid];
    auto &b = vlist[bid];

    sol->solve(*x, *b);

    return 0;
}

int XAMG_solver_get_convergence_info(int *conv_info, uint16_t nv, XAMG_solver_id sid) {
    helper::check_rhs(nv);
    auto &sol = slist[sid];

    uint16_t i;
    for (i = 0; i < nv; i++)
        conv_info[i] = sol->stats.if_converged[i];
    return 0;
}

int XAMG_solver_destroy(XAMG_solver_id sid) {
    if (slist[sid] != nullptr) {
        //        delete slist[sid];
        slist[sid] = nullptr;
    }
    return 0;
}

/////////

int XAMG_barrier() {
    XAMG::mpi::barrier();
    return 0;
}

double XAMG_timer() {
    return XAMG::sys::timer();
}

int XAMG_get_nv() {
    return CONST_NV;
}

int XAMG_init(int argc, char **argv, char *conf) {
    XAMG::init(argc, argv, conf);
    return 0;
}

int XAMG_finalize() {
    XAMG::finalize();

    for (auto p : param_list) {
        delete p;
    }

    for (auto v : vlist) {
        delete v;
    }

    for (auto mcsr : mcsrlist) {
        delete mcsr;
    }

    for (auto m : mlist) {
        delete m;
    }

    for (auto s : slist) {
        //        delete s;
        s = nullptr;
    }
    return 0;
}

} //!!! end of C-source
