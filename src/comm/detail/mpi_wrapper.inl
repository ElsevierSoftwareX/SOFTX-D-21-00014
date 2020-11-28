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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <algorithm>

#ifdef XAMG_SEPARATE_OBJECT
#include <sys/numaconf.h>
#endif

namespace XAMG {
namespace mpi {

void set_exec_id() {
    if (!id.gl_proc) {
        srand(time(NULL));
        id.exec_id = rand() % 100000;
    }
    mpi::bcast<uint32_t>(&(id.exec_id), 1, 0, mpi::GLOBAL);
}

scope intra_layer_comm(segment::hierarchy layer) {
    scope comm = SELF;
    switch (layer) {
    case segment::NODE: {
        comm = INTRA_NODE;
        break;
    }
    case segment::NUMA: {
        comm = INTRA_NUMA;
        break;
    }
    case segment::CORE: {
        comm = SELF;
        break;
    }
    default: {
        assert(0);
        break;
    }
    }

    return comm;
}

scope cross_layer_comm(segment::hierarchy layer) {
    scope comm = SELF;
    switch (layer) {
    case segment::NODE: {
        comm = CROSS_NODE;
        break;
    }
    case segment::NUMA: {
        comm = CROSS_NUMA;
        break;
    }
    case segment::CORE: {
        comm = INTRA_NUMA;
        break;
    }
    default: {
        assert(0);
        break;
    }
    }

    return comm;
}

//struct comm_pool {
//    int proc, nprocs;
//    scope comm;
//    bool master_proc;
//
//    comm_pool(const scope _comm) : comm(_comm), proc(-1), nprocs(-1), master_proc(false) {
//        assign();
//
//        if (!proc)
//            master_proc = true;
//    }
//
//    void assign() {
//        switch (comm) {
//        case GLOBAL: {
//            proc = id.gl_proc;
//            nprocs = id.gl_nprocs;
//            break;
//        }
//        case INTRA_NODE: {
//            proc = id.nd_core;
//            nprocs = id.nd_ncores;
//            break;
//        }
//        case INTRA_NUMA: {
//            proc = id.nm_core;
//            nprocs = id.nm_ncores;
//            break;
//        }
//        case CROSS_NODE: {
//            proc = id.gl_node;
//            nprocs = id.gl_nnodes;
//            break;
//        }
//        case CROSS_NUMA: {
//            proc = id.nd_numa;
//            nprocs = id.nd_nnumas;
//            break;
//        }
//        case SELF: {
//            proc = 0;
//            nprocs = 1;
//            break;
//        }
//        default: {
//            assert(0);
//            break;
//        }
//        }
//    }
//};
//}

struct MPI_ID {
    MPI_Comm global_comm;
    MPI_Comm cross_node_comm; // among the master node-processes per comm
    MPI_Comm intra_node_comm; // among the processes inside the node
    MPI_Comm cross_numa_comm; // among the master numa-processes per node
    MPI_Comm intra_numa_comm; // among the processes inside the numa
    MPI_Comm self_comm;       // self communicator

    MPI_Comm &get_comm(scope comm) {
        if (comm == GLOBAL) {
            return global_comm;
        } else if (comm == CROSS_NODE) {
            return cross_node_comm;
        } else if (comm == INTRA_NODE) {
            return intra_node_comm;
        } else if (comm == CROSS_NUMA) {
            return cross_numa_comm;
        } else if (comm == INTRA_NUMA) {
            return intra_numa_comm;
        } else if (comm == SELF) {
            return self_comm;
        } else {
            assert(0 && "unknown comm type");
        }

        return global_comm;
    }
};

struct get_mpi_type {
    template <typename T>
    MPI_Datatype type();
};

get_mpi_type mpi_type;

template <>
MPI_Datatype get_mpi_type::type<int>() {
    return MPI_INT;
};
template <>
MPI_Datatype get_mpi_type::type<uint8_t>() {
    return MPI_UNSIGNED_CHAR;
};
template <>
MPI_Datatype get_mpi_type::type<uint16_t>() {
    return MPI_UNSIGNED_SHORT;
};
template <>
MPI_Datatype get_mpi_type::type<uint32_t>() {
    return MPI_UNSIGNED;
};
template <>
MPI_Datatype get_mpi_type::type<uint64_t>() {
    return MPI_UNSIGNED_LONG;
};

template <>
MPI_Datatype get_mpi_type::type<float32_t>() {
    return MPI_FLOAT;
};
template <>
MPI_Datatype get_mpi_type::type<float64_t>() {
    return MPI_DOUBLE;
};

uint32_t get_host_id() {
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int processor_name_len;
    MPI_Get_processor_name(processor_name, &processor_name_len);
    uint32_t local_hash = XAMG::misc::hash(processor_name);
    std::vector<uint32_t> hosts_hash(id.gl_nprocs);
    // XAMG::out << XAMG::DBG << XAMG::ALLRANKS << processor_name << " " << local_hash << std::endl;
    // XAMG::out << XAMG::DBG << ALLRANKS << "size = "<< id.gl_nprocs << std::endl;

    allgather<uint32_t>(&local_hash, 1, &hosts_hash[0], 1);
    std::sort(hosts_hash.begin(), hosts_hash.end());

    uint32_t host_id = 0;
    uint32_t val = hosts_hash[0];
    for (auto hash : hosts_hash) {
        if (hash != val) {
            val = hash;
            host_id++;
        }
        if (hash == local_hash)
            break;
    }

    return host_id;
}

void setup_node_comm_layer(const uint32_t host_id) {
    //  Node comm layer:
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);

    int color = host_id;
    int key = id.gl_proc;
    MPI_Comm_split(mpi_id.global_comm, color, key, &(mpi_id.intra_node_comm));
    assert(mpi_id.intra_node_comm != MPI_COMM_NULL);

    MPI_Comm_size(mpi_id.intra_node_comm, &(id.nd_ncores));
    MPI_Comm_rank(mpi_id.intra_node_comm, &(id.nd_core));

    /////////

    if (!id.nd_core)
        color = 0;
    else
        color = MPI_UNDEFINED;
    MPI_Comm_split(mpi_id.global_comm, color, key, &(mpi_id.cross_node_comm));

    if (!id.nd_core) {
        assert(mpi_id.cross_node_comm != MPI_COMM_NULL);

        MPI_Comm_size(mpi_id.cross_node_comm, &(id.gl_nnodes));
        MPI_Comm_rank(mpi_id.cross_node_comm, &(id.gl_node));
    }
    //    else
    //        id.gl_node = -1;

    MPI_Bcast(&id.gl_node, 1, MPI_INT, 0, mpi_id.intra_node_comm);
    MPI_Bcast(&id.gl_nnodes, 1, MPI_INT, 0, mpi_id.intra_node_comm);
}

void destroy_node_comm_layer() {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);

    if (mpi_id.intra_node_comm != MPI_COMM_NULL)
        MPI_Comm_free(&mpi_id.intra_node_comm);

    if (mpi_id.cross_node_comm != MPI_COMM_NULL)
        MPI_Comm_free(&mpi_id.cross_node_comm);
}

void setup_numa_comm_layer(const uint32_t numa_id) {
    //  Numa comm layer:
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);

    int color = numa_id;
    int key = id.nd_core;
    MPI_Comm_split(mpi_id.intra_node_comm, color, key, &(mpi_id.intra_numa_comm));
    assert(mpi_id.intra_numa_comm != MPI_COMM_NULL);

    MPI_Comm_size(mpi_id.intra_numa_comm, &(id.nm_ncores));
    MPI_Comm_rank(mpi_id.intra_numa_comm, &(id.nm_core));

    /////////

    if (!id.nm_core)
        color = 0;
    else
        color = MPI_UNDEFINED;
    MPI_Comm_split(mpi_id.intra_node_comm, color, key, &(mpi_id.cross_numa_comm));

    if (!id.nm_core) {
        assert(mpi_id.cross_numa_comm != MPI_COMM_NULL);

        MPI_Comm_size(mpi_id.cross_numa_comm, &(id.nd_nnumas));
        MPI_Comm_rank(mpi_id.cross_numa_comm, &(id.nd_numa));
    }
    //    else
    //        id.nd_numa = -1;

    MPI_Bcast(&id.nd_numa, 1, MPI_INT, 0, mpi_id.intra_numa_comm);
    MPI_Bcast(&id.nd_nnumas, 1, MPI_INT, 0, mpi_id.intra_numa_comm);
}

void destroy_numa_comm_layer() {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);

    if (mpi_id.intra_numa_comm != MPI_COMM_NULL)
        MPI_Comm_free(&mpi_id.intra_numa_comm);

    if (mpi_id.cross_numa_comm != MPI_COMM_NULL)
        MPI_Comm_free(&mpi_id.cross_numa_comm);
}

void init(int argc, char **argv, const sys::numa_conf &conf) {
    id.mpi_details = new MPI_ID;

    int if_initialized;
    MPI_Initialized(&if_initialized);
    if (!if_initialized)
        MPI_Init(&argc, &argv);

    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    mpi_id.global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(mpi_id.global_comm, &id.gl_nprocs);
    MPI_Comm_rank(mpi_id.global_comm, &id.gl_proc);

    uint32_t numa_id, gpu_id;
    uint32_t host_id, core_id;

    id.nd_nnumas = conf.nnumas;
    id.nd_ncores = (conf.ncores * conf.nnumas);
    id.nm_ncores = conf.ncores;
    if (!conf.noautoconf) {
        host_id = get_host_id();
    } else {
        host_id = id.gl_proc / id.nd_ncores;
    }
    if (conf.core == -1) {
        conf.core = id.gl_proc % id.nd_ncores;
        core_id = conf.core % id.nm_ncores;
    } else {
        core_id = ((uint32_t)conf.core % id.nm_ncores);
    }
    gpu_id = conf.gpu_by_core(conf.core);
    numa_id = conf.numa_by_core(conf.core);

    /////////

    setup_node_comm_layer(host_id);
    setup_numa_comm_layer(numa_id);
    MPI_Comm_dup(MPI_COMM_SELF, &(mpi_id.self_comm));

    usleep(id.gl_proc * 1000);
    XAMG::out << XAMG::DBG << XAMG::ALLRANKS << XAMG::LOG << "Proc : " << id.gl_proc << " host_id "
              << host_id << " numa_id " << numa_id << " core_id " << core_id << "  ||| "
              << " gl_node " << id.gl_node << " nd_core " << id.nd_core << " nd_numa " << id.nd_numa
              << " nm_core " << id.nm_core << std::endl;
    XAMG::mpi::barrier();
    usleep(1000);
}

void finalize() {
    int if_finalized;
    MPI_Finalized(&if_finalized);

    if (!if_finalized) {
        destroy_numa_comm_layer();
        destroy_node_comm_layer();
        MPI_Finalize();
    }

    delete ((MPI_ID *)id.mpi_details);
}

void abort(const std::string &str) {
    (void)str;
    MPI_Abort(MPI_COMM_WORLD, 1);
}

void wait(token &tok) {
    MPI_Wait(token2req(tok), MPI_STATUS_IGNORE);
}

int waitany(tokens &toks) {
    int idx;
    MPI_Waitany(get_num_tokens(toks), tokens2reqptr(toks), &idx, MPI_STATUSES_IGNORE);
    return idx;
}

void waitall(tokens &toks) {
    MPI_Waitall(get_num_tokens(toks), tokens2reqptr(toks), MPI_STATUSES_IGNORE);
}

bool test(token &tok) {
    int flag;
    MPI_Test(token2req(tok), &flag, MPI_STATUS_IGNORE);
    return flag != 0;
}

bool testany(tokens &toks, int &idx) {
    int flag;
    MPI_Testany(get_num_tokens(toks), tokens2reqptr(toks), &idx, &flag, MPI_STATUSES_IGNORE);
    return flag != 0;
}

bool testall(tokens &toks) {
    int flag;
    MPI_Testall(get_num_tokens(toks), tokens2reqptr(toks), &flag, MPI_STATUSES_IGNORE);
    return flag != 0;
}

static inline void cycle_over_message_parts(void *ptr, size_t size, size_t type_size, size_t nparts,
                                            std::function<void(char *, size_t, int i)> actor) {
    char *p = (char *)ptr;
    size_t rest = size % MAXMPIMSGSIZE;
    for (int i = 0; i < (int)nparts; i++) {
        size_t portion = (size >= MAXMPIMSGSIZE ? MAXMPIMSGSIZE : rest);
        assert((int)portion >= 0);
        actor(p, portion, i);
        p += portion * type_size;
        size -= portion;
    }
    assert(size == 0);
}

template <typename T>
void isend(void *ptr, size_t size, uint32_t rank, token &tok, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    assert((int)size >= 0);
    MPI_Isend(ptr, size, mpi_type.type<T>(), rank, tag, mpi_id.get_comm(comm), token2req(tok));
}

template <typename T>
void isend_bigsize(void *ptr, size_t size, uint32_t rank, tokens &toks, size_t start_tok,
                   size_t ntoks, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    assert(num_parts_for_message(size) == (int)ntoks);
    cycle_over_message_parts(ptr, size, sizeof(T), num_parts_for_message(size),
                             [&](char *p, size_t portion, int i) {
                                 MPI_Isend(p, (int)portion, mpi_type.type<T>(), rank, tag,
                                           mpi_id.get_comm(comm), token2req(toks[start_tok + i]));
                             });
}

template <typename T>
void send(void *ptr, size_t size, uint32_t rank, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    assert((int)size >= 0);
    MPI_Send(ptr, size, mpi_type.type<T>(), rank, tag, mpi_id.get_comm(comm));
}

template <typename T>
void send_bigsize(void *ptr, size_t size, uint32_t rank, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    cycle_over_message_parts(
        ptr, size, sizeof(T), num_parts_for_message(size), [&](char *p, size_t portion, int i) {
            (void)i;
            MPI_Send(p, (int)portion, mpi_type.type<T>(), rank, tag, mpi_id.get_comm(comm));
        });
}

template <typename T>
void irecv(void *ptr, size_t size, uint32_t rank, token &tok, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    assert((int)size >= 0);
    MPI_Irecv(ptr, size, mpi_type.type<T>(), rank, tag, mpi_id.get_comm(comm), token2req(tok));
}

template <typename T>
void irecv_bigsize(void *ptr, size_t size, uint32_t rank, tokens &toks, size_t start_tok,
                   size_t ntoks, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    assert(num_parts_for_message(size) == (int)ntoks);
    cycle_over_message_parts(ptr, size, sizeof(T), num_parts_for_message(size),
                             [&](char *p, size_t portion, int i) {
                                 MPI_Irecv(p, (int)portion, mpi_type.type<T>(), rank, tag,
                                           mpi_id.get_comm(comm), token2req(toks[start_tok + i]));
                             });
}

template <typename T>
void recv(void *ptr, size_t size, uint32_t rank, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    assert((int)size >= 0);
    MPI_Recv(ptr, size, mpi_type.type<T>(), rank, tag, mpi_id.get_comm(comm), MPI_STATUS_IGNORE);
}

template <typename T>
void recv_bigsize(void *ptr, size_t size, uint32_t rank, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    cycle_over_message_parts(ptr, size, sizeof(T), num_parts_for_message(size),
                             [&](char *p, size_t portion, int i) {
                                 (void)i;
                                 MPI_Recv(p, (int)portion, mpi_type.type<T>(), rank, tag,
                                          mpi_id.get_comm(comm), MPI_STATUS_IGNORE);
                             });
}

template <typename T>
void probe(uint32_t rank, size_t &size, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Status stat;
    MPI_Probe(rank, tag, mpi_id.get_comm(comm), &stat);
    int isize;
    MPI_Get_count(&stat, mpi_type.type<T>(), &isize);
    size = (uint64_t)isize;
}

template <typename T>
void probe_any_source(uint32_t &rank, size_t &size, scope comm, int tag) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Status stat;
    MPI_Probe(MPI_ANY_SOURCE, tag, mpi_id.get_comm(comm), &stat);
    rank = stat.MPI_SOURCE;
    int isize;
    MPI_Get_count(&stat, mpi_type.type<T>(), &isize);
    size = (uint64_t)isize;
}

void barrier(scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Barrier(mpi_id.get_comm(comm));
}

template <typename T>
void allreduce_sum(T *in, T *out, size_t size, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Allreduce(in, out, size, mpi_type.type<T>(), MPI_SUM, mpi_id.get_comm(comm));
}

template <typename T>
void iallreduce_sum_init(T *in, T *out, size_t size, token &tok, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Iallreduce(in, out, size, mpi_type.type<T>(), MPI_SUM, mpi_id.get_comm(comm),
                   token2req(tok));
}

template <typename T>
void allreduce_min(T *in, T *out, size_t size, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Allreduce(in, out, size, mpi_type.type<T>(), MPI_MIN, mpi_id.get_comm(comm));
}

template <typename T>
void allreduce_max(T *in, T *out, size_t size, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Allreduce(in, out, size, mpi_type.type<T>(), MPI_MAX, mpi_id.get_comm(comm));
}

template <typename T>
void iallreduce_max_init(T *in, T *out, size_t size, token &tok, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Iallreduce(in, out, size, mpi_type.type<T>(), MPI_MAX, mpi_id.get_comm(comm),
                   token2req(tok));
}

template <typename T>
void alltoall(T *in, size_t size_in, T *out, size_t size_out, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Alltoall(in, size_in, mpi_type.type<T>(), out, size_out, mpi_type.type<T>(),
                 mpi_id.get_comm(comm));
}

template <typename T>
void gather(T *in, size_t size_in, T *out, size_t size_out, uint32_t root, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Gather(in, size_in, mpi_type.type<T>(), out, size_out, mpi_type.type<T>(), root,
               mpi_id.get_comm(comm));
}

template <typename T>
void gatherv(T *in, size_t size_in, T *out, int *size_out, int *offset, uint32_t root, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Gatherv(in, size_in, mpi_type.type<T>(), out, size_out, offset, mpi_type.type<T>(), root,
                mpi_id.get_comm(comm));
}

template <typename T>
void allgather(T *in, size_t size_in, T *out, size_t size_out, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Allgather(in, size_in, mpi_type.type<T>(), out, size_out, mpi_type.type<T>(),
                  mpi_id.get_comm(comm));
}

template <typename T>
void allgatherv(T *in, size_t size_in, T *out, int *size_out, int *offset, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Allgatherv(in, size_in, mpi_type.type<T>(), out, size_out, offset, mpi_type.type<T>(),
                   mpi_id.get_comm(comm));
}

template <typename T>
void scatter(T *in, size_t size_in, T *out, size_t size_out, uint32_t root, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Scatter(in, size_in, mpi_type.type<T>(), out, size_out, mpi_type.type<T>(), root,
                mpi_id.get_comm(comm));
}

template <typename T>
void bcast(T *in, size_t size, int root, scope comm) {
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    MPI_Bcast(in, size, mpi_type.type<T>(), root, mpi_id.get_comm(comm));
}

///////////////////

template <typename T>
void bcast(std::vector<T> &in, int root, scope comm) {
    assert(0);
    MPI_ID &mpi_id = *((MPI_ID *)id.mpi_details);
    uint32_t vec_size = in.size();
    mpi::bcast<uint32_t>(&vec_size, 1, root, comm);
    if (!root)
        in.resize(vec_size);
    mpi::bcast<T>(in.data(), vec_size, root, comm);
}

} // namespace mpi
} // namespace XAMG

void *ID::get_comm() {
    XAMG::mpi::MPI_ID &mpi_id = *((XAMG::mpi::MPI_ID *)id.mpi_details);
    return ((void *)&mpi_id.global_comm);
};

void *ID::get_comm(const XAMG::segment::hierarchy layer) {
    XAMG::mpi::MPI_ID &mpi_id = *((XAMG::mpi::MPI_ID *)id.mpi_details);
    if (layer == XAMG::segment::NODE) {
        return ((void *)&mpi_id.cross_node_comm);
    } else if (layer == XAMG::segment::NUMA) {
        return ((void *)&mpi_id.cross_numa_comm);
    } else if (layer == XAMG::segment::CORE) {
        return ((void *)&mpi_id.intra_numa_comm);
    } else {
        assert(0);
    }

    return nullptr;
};
