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

#include "posix_shm/shm.h"

extern ID id;

namespace XAMG {
namespace comm {

struct atomic_sync { // non-blocking processes synchronization
    vector::vector shared_vec;
    volatile int *shared;
    int i;
    mpi::comm_pool comm_group;

    atomic_sync(const segment::hierarchy layer)
        : shared_vec(mem::SHARED, sharing_mode_by_intra_layer(layer)), shared(nullptr), i(0),
          comm_group(mpi::intra_layer_comm(layer)) {}

    void alloc() {
        assert((shared == nullptr) && "Reallocation of atomic_sync objects is not allowed");
        shared = (int *)shared_vec.alloc_and_get_aligned_ptr<uint32_t>(2);
        if (comm_group.master_proc)
            shared[0] = shared[1] = 0;
    }

    ~atomic_sync() {
        // shm_free(shared);
        shared = nullptr;
    }
    //  TODO: check if copy-constructor must be deleted
    //    atomic_sync(const atomic_sync &that) = delete;
    //    atomic_sync& operator=(const atomic_sync &that) = delete;
};

struct barrier : atomic_sync { // non-blocking barrier over shared memory
    barrier(const segment::hierarchy layer) : atomic_sync(layer) {}

    void init() {
        // XAMG::out << XAMG::ALLRANKS << "Barrier.init start" << std::endl;

        int ii = __sync_add_and_fetch(&(shared[i]), 1);
        if (ii == comm_group.nprocs)
            __sync_sub_and_fetch(&(shared[i]), comm_group.nprocs);

        // XAMG::out << XAMG::ALLRANKS << "Barrier.init end" << std::endl;
    }

    void wait() {
        // XAMG::out << XAMG::ALLRANKS << "Barrier.wait start" << std::endl;
        while (shared[i] > 0)
            ;
        i ^= 1;
        // XAMG::out << XAMG::ALLRANKS << "Barrier.wait end" << std::endl;
    }
};

struct sync_all : atomic_sync { // master process has reached certain point
    sync_all(const segment::hierarchy layer) : atomic_sync(layer) {}
    // ~sync_all() {}

    void init() {
        // XAMG::out << XAMG::ALLRANKS << "Syncall.init start" << std::endl;
        if (comm_group.master_proc) {
            assert(shared[i] == 0);
        }
        if (comm_group.master_proc)
            int ii = __sync_add_and_fetch(&(shared[i]), comm_group.nprocs);
        // XAMG::out << XAMG::ALLRANKS << "Syncall.init end" << std::endl;
    }

    void wait() {
        // XAMG::out << XAMG::ALLRANKS << "Syncall.wait start" << std::endl;
        while (!shared[i])
            ;
        int ii = __sync_sub_and_fetch(&(shared[i]), 1);
        assert(ii >= 0);
        i ^= 1;
        // XAMG::out << XAMG::ALLRANKS << "Syncall.wait end" << std::endl;
    }
};

struct sync_one : atomic_sync { // all processes have reached certain point
    sync_one(const segment::hierarchy layer) : atomic_sync(layer) {}
    // ~sync_one() {}

    void init() {
        // XAMG::out << XAMG::ALLRANKS << "Syncone.init start" << std::endl;
        int ii = __sync_add_and_fetch(&(shared[i]), 1);
        // XAMG::out << XAMG::ALLRANKS << "Pushing to " << i
        //           << " : Result " << ii << " Check " << shared[i] << std::endl;

        assert(ii <= comm_group.nprocs);
        if (!comm_group.master_proc)
            i ^= 1;
    }

    void wait() {
        // XAMG::out << XAMG::ALLRANKS << "Syncone.wait start" << id.gl_proc << std::endl;
        // XAMG::out << XAMG::ALLRANKS << "i = " << i << std::endl;
        if (comm_group.master_proc) {
            while (shared[i] < comm_group.nprocs)
                ;
            assert(shared[i] == comm_group.nprocs);
            uint32_t ii = __sync_sub_and_fetch(&(shared[i]), comm_group.nprocs);
            assert(ii == 0);

            i ^= 1;
        }
        // XAMG::out << XAMG::ALLRANKS << "Syncone.wait end" << id.gl_proc << std::endl;
    }
};

} // namespace comm
} // namespace XAMG
