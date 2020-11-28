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

#include "xamg_types.h"
#include "xamg.h"
#include "shm_sync.h"

extern ID id;

namespace XAMG {
namespace comm {

template <typename T, const uint16_t NV>
struct allreduce : XAMG::mpi::token_ptr {
    sync_one shm_syncone;
    sync_all shm_syncall;
    barrier shm_barrier;
    bool async_flag;

    T *shared;
    T *local;
    vector::vector shared_vec;
    vector::vector local_vec;

    uint64_t buffer_size;
    uint32_t max_buffer_size;
    mpi::comm_pool comm_group;

    allreduce(segment::hierarchy layer = segment::NODE)
        : shm_syncone(layer), shm_syncall(layer), shm_barrier(layer), async_flag(false),
          shared(nullptr), local(nullptr), shared_vec(mem::SHARED, mem::NODE),
          local_vec(mem::LOCAL), buffer_size(0), max_buffer_size(0),
          comm_group(mpi::intra_layer_comm(layer)) {
        assert(layer == segment::NODE);
        mpi::barrier(comm_group.comm);
        shm_syncone.alloc();
        shm_syncall.alloc();
        shm_barrier.alloc();
        mpi::barrier(comm_group.comm);
    }

    void alloc(const uint32_t max_buffer_size_) {
        max_buffer_size = max_buffer_size_;
        shared = shared_vec.alloc_and_get_aligned_ptr<T>(max_buffer_size * comm_group.nprocs, NV);
        local = local_vec.alloc_and_get_aligned_ptr<T>(max_buffer_size, NV);
        shm_barrier.init();
    }

    void push_array(const T *elem, const uint64_t size) {
        assert(buffer_size + size <= max_buffer_size);
        memcpy(local + buffer_size * NV, elem, size * NV * sizeof(T));
        buffer_size += size;
    }

    void pull_array(T *elem, const uint64_t size) {
        assert(buffer_size >= size);
        buffer_size -= size;
        memcpy(elem, local + buffer_size * NV, size * NV * sizeof(T));
    }

    void push_vector(const vector::vector &vec) {
        vec.check(vec.initialized);
        assert(vec.type_hash == typeid(T).hash_code());
        assert(vec.nv == NV);

        auto ptr = vec.get_aligned_ptr<T>();
        push_array(ptr, vec.size);
    }

    void pull_vector(vector::vector &vec) {
        vec.check(vec.allocated);
        assert(vec.type_hash == typeid(T).hash_code());
        assert(vec.nv == NV);

        auto ptr = vec.get_aligned_ptr<T>();
        pull_array(ptr, vec.size);
        vec.if_initialized = true;

        vec.if_zero = true;
        for (uint32_t i = 0; i < vec.size * vec.nv; ++i) {
            if (ptr[i] != (T)0)
                vec.if_zero = false;
        }
    }

    ///////////////////

    void init() {
        shm_barrier.wait();

        uint32_t shift = comm_group.proc * buffer_size * NV;
        // copy from local buffer to shared
        for (uint32_t l = 0; l < buffer_size; ++l)
            for (uint16_t nv = 0; nv < NV; ++nv)
                shared[shift + l * NV + nv] = local[l * NV + nv];

        shm_syncone.init();

        if (comm_group.master_proc) {
            shm_syncone.wait();

            for (int np = 1; np < comm_group.nprocs; ++np) {
                uint32_t shift0 = np * buffer_size * NV;

                for (uint32_t l = 0; l < buffer_size; ++l)
                    for (uint16_t nv = 0; nv < NV; ++nv)
                        // shared[l*NV + nv] += shared[shift + l*NV + nv];
                        local[l * NV + nv] += shared[shift0 + l * NV + nv];
            }
        }
    }

    void process_sync_action() {
        async_flag = false;
        if (comm_group.master_proc) {
            mpi::allreduce_sum<T>(local, shared, buffer_size * NV, mpi::CROSS_NODE);
            shm_syncall.init();
        }
        perf.allreduce(1);
    }

    void process_async_action() {
        async_flag = true;
        if (comm_group.master_proc)
            mpi::iallreduce_sum_init<T>(local, shared, buffer_size * NV, get_token(),
                                        mpi::CROSS_NODE);
        perf.iallreduce(1);
    }

    void wait() {
        if ((async_flag) && (comm_group.master_proc)) {
            mpi::wait(get_token());
            shm_syncall.init();
        }

        shm_syncall.wait();
        memcpy(local, shared, buffer_size * NV * sizeof(T));
        shm_barrier.init();
    }
};

} // namespace comm
} // namespace XAMG
