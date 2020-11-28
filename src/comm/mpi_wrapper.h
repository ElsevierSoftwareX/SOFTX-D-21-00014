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

#include "misc/misc.h"
#include "comm/mpi_token.h"

namespace XAMG {
namespace sys {
struct numa_conf;
}
} // namespace XAMG

namespace XAMG {
namespace mpi {

constexpr size_t TWO_POW_31_MINUS_1 = 2147483647;
constexpr size_t MAXMPIMSGSIZE = TWO_POW_31_MINUS_1;

enum scope { GLOBAL, CROSS_NODE, INTRA_NODE, CROSS_NUMA, INTRA_NUMA, SELF };

void set_exec_id();

scope intra_layer_comm(segment::hierarchy layer);
scope cross_layer_comm(segment::hierarchy layer);

void init(int argc, char **argv, const sys::numa_conf &conf);
void finalize();
void abort(const std::string &str);

void wait(token &tok);
int waitany(tokens &toks);
void waitall(tokens &toks);

bool test(token &tok);
bool testany(tokens &toks, int &idx);
bool testall(tokens &toks);

template <typename T>
void isend(void *ptr, size_t size, uint32_t rank, token &tok, scope comm = GLOBAL, int tag = 0);

template <typename T>
void isend_bigsize(void *ptr, size_t size, uint32_t rank, tokens &toks, size_t start_tok,
                   size_t ntoks, scope comm = GLOBAL, int tag = 0);

template <typename T>
void send(void *ptr, size_t size, uint32_t rank, scope comm = GLOBAL, int tag = 0);

template <typename T>
void send_bigsize(void *ptr, size_t size, uint32_t rank, scope comm = GLOBAL, int tag = 0);

template <typename T>
void irecv(void *ptr, size_t size, uint32_t rank, token &tok, scope comm = GLOBAL, int tag = 0);

template <typename T>
void irecv_bigsize(void *ptr, size_t size, uint32_t rank, tokens &toks, size_t start_tok,
                   size_t ntoks, scope comm = GLOBAL, int tag = 0);

template <typename T>
void recv(void *ptr, size_t size, uint32_t rank, scope comm = GLOBAL, int tag = 0);

template <typename T>
void recv_bigsize(void *ptr, size_t size, uint32_t rank, scope comm = GLOBAL, int tag = 0);

template <typename T>
void probe(uint32_t rank, size_t &size, scope comm = GLOBAL, int tag = 0);

template <typename T>
void probe_any_source(uint32_t &rank, size_t &size, scope comm = GLOBAL, int tag = 0);

void barrier(scope comm = GLOBAL);

template <typename T>
void allreduce_sum(T *in, T *out, size_t size, scope comm = GLOBAL);

template <typename T>
void iallreduce_sum_init(T *in, T *out, size_t size, token &tok, scope comm = GLOBAL);

template <typename T>
void allreduce_min(T *in, T *out, size_t size, scope comm = GLOBAL);

template <typename T>
void allreduce_max(T *in, T *out, size_t size, scope comm = GLOBAL);

template <typename T>
void iallreduce_max_init(T *in, T *out, size_t size, token &tok, scope comm = GLOBAL);

template <typename T>
void alltoall(T *in, size_t size_in, T *out, size_t size_out, scope comm = GLOBAL);

template <typename T>
void gather(T *in, size_t size_in, T *out, size_t size_out, uint32_t root, scope comm = GLOBAL);

template <typename T>
void gatherv(T *in, size_t size_in, T *out, int *size_out, int *offset, uint32_t root,
             scope comm = GLOBAL);

template <typename T>
void allgather(T *in, size_t size_in, T *out, size_t size_out, scope comm = GLOBAL);

template <typename T>
void allgatherv(T *in, size_t size_in, T *out, int *size_out, int *offset, scope comm = GLOBAL);

template <typename T>
void scatter(T *in, size_t size_in, T *out, size_t size_out, uint32_t root, scope comm = GLOBAL);

template <typename T>
void bcast(T *in, size_t size, int root, scope comm = GLOBAL);

static inline int num_parts_for_message(size_t size) {
    return size ? ((size - 1) / MAXMPIMSGSIZE + 1) : 1;
}

///////////////////

struct comm_pool {
    scope comm;
    int proc, nprocs;
    bool master_proc;
    bool member_proc;

    comm_pool(const scope _comm)
        : comm(_comm), proc(-1), nprocs(-1), master_proc(false), member_proc(false) {
        assign();
        if (!proc)
            master_proc = true;
    }

    void assign() {
        switch (comm) {
        case GLOBAL: {
            proc = id.gl_proc;
            nprocs = id.gl_nprocs;
            member_proc = true;
            break;
        }
        case INTRA_NODE: {
            proc = id.nd_core;
            nprocs = id.nd_ncores;
            member_proc = true;
            break;
        }
        case INTRA_NUMA: {
            proc = id.nm_core;
            nprocs = id.nm_ncores;
            member_proc = true;
            break;
        }
        case CROSS_NODE: {
            proc = id.gl_node;
            nprocs = id.gl_nnodes;
            if (!id.nd_core)
                member_proc = true;
            break;
        }
        case CROSS_NUMA: {
            proc = id.nd_numa;
            nprocs = id.nd_nnumas;
            if (!id.nm_core)
                member_proc = true;
            break;
        }
        case SELF: {
            proc = 0;
            nprocs = 1;
            member_proc = true;
            break;
        }
        default: {
            assert(0);
            break;
        }
        }
    }
};

// template <typename T>
// void bcast(std::vector<T> &in, int root, scope comm = GLOBAL);

} // namespace mpi
} // namespace XAMG
