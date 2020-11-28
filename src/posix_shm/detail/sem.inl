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

namespace XAMG {
namespace shm {

struct sem_object {
    std::string handle;
    sem_t *ptr;

    sem_object(const uint32_t val, mem::sharing sharing_mode) { //mpi::scope comm) {
        mpi::comm_pool comm_group(mpi::intra_layer_comm(intra_layer_by_sharing_mode(sharing_mode)));
        //        assert((comm_group.comm == mpi::INTRA_NODE) || (comm_group.comm == mpi::INTRA_NUMA));

        int sem_id;
        switch (sharing_mode) {
        case mem::NODE: {
            sem_id = id.node_sem_id;
            ++id.node_sem_id;
            break;
        }
        case mem::NUMA: {
            sem_id = id.numa_sem_id;
            ++id.numa_sem_id;
            break;
        }
        default: {
            assert(0);
            break;
        }
        }

        handle = "/xamg_sem." + std::to_string(id.exec_id) + ".nd." + std::to_string(id.gl_node) +
                 ".nm." + std::to_string(id.nd_numa) + ".id." + std::to_string(sem_id);

        if (!comm_group.master_proc)
            mpi::barrier(comm_group.comm);

        ptr = sem_open(handle.c_str(), O_CREAT | O_RDWR, 0777, val);

        if (comm_group.master_proc) {
            int sem_val;
            sem_getvalue(ptr, &sem_val);

            while (sem_val) {
                sem_wait(ptr);
                sem_val--;
            }

            while (sem_val < (int)val) {
                sem_post(ptr);
                sem_val++;
            }
        }

        if (comm_group.master_proc)
            mpi::barrier(comm_group.comm);
    }

    ~sem_object() {
        sem_close(ptr);
        sem_unlink(handle.c_str());
    }

    void post() { sem_post(ptr); }
    void wait() { sem_wait(ptr); }
};

} // namespace shm
} // namespace XAMG
