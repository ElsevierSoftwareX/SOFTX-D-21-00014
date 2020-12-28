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

#include "comm/mpi_token.h"
#include "comm/mpi_wrapper.h"

#include "misc/misc.h"

namespace XAMG {
namespace part {

struct part_layer {

    std::vector<uint64_t> block_indx;
    std::vector<uint64_t> block_size;
    uint32_t nblocks;

    void get_part_layer(uint64_t proc_size, const segment::hierarchy layer) {
        mpi::comm_pool comm_group(mpi::cross_layer_comm(layer));

        nblocks = comm_group.nprocs;
        block_size.resize(nblocks);
        block_indx.resize(nblocks + 1);

        mpi::allgather<uint64_t>(&proc_size, 1, block_size.data(), 1, comm_group.comm);

        block_indx[0] = 0;
        for (uint32_t i = 0; i < (uint32_t)comm_group.nprocs; ++i)
            block_indx[i + 1] = block_indx[i] + block_size[i];
    }

    void segment_part_layer(uint64_t size, const segment::hierarchy layer) {
        mpi::comm_pool comm_group(mpi::cross_layer_comm(layer));

        nblocks = comm_group.nprocs;
        block_size.resize(nblocks);
        block_indx.resize(nblocks + 1);

        mpi::bcast<uint64_t>(&size, 1, 0, comm_group.comm);
        uint64_t proc_size = size / nblocks;
        for (uint32_t nb = 0; nb < nblocks - 1; ++nb)
            block_size[nb] = proc_size;
        block_size[nblocks - 1] = (size - (nblocks - 1) * proc_size);

        block_indx[0] = 0;
        for (uint32_t i = 0; i < (uint32_t)comm_group.nprocs; ++i)
            block_indx[i + 1] = block_indx[i] + block_size[i];
    }

    void share_part_layer(const segment::hierarchy layer) {
        mpi::comm_pool comm_group(mpi::cross_layer_comm(layer));

        mpi::bcast<uint32_t>(&nblocks, 1, 0, comm_group.comm);

        if (!comm_group.master_proc) {
            block_size.resize(nblocks);
            block_indx.resize(nblocks + 1);
        }

        mpi::bcast<uint64_t>(block_size.data(), nblocks, 0, comm_group.comm);
        mpi::bcast<uint64_t>(block_indx.data(), nblocks + 1, 0, comm_group.comm);
    }
};

struct part {

    part_layer node_layer;
    part_layer numa_layer;
    part_layer core_layer;

    part() {}
    part(const uint64_t &size, const uint32_t nblocks) {
        node_layer.nblocks = nblocks;
        node_layer.block_size.resize(nblocks);
        node_layer.block_indx.resize(nblocks + 1);

        uint64_t block_size = size / nblocks;
        for (uint32_t nb = 0; nb < nblocks - 1; ++nb)
            node_layer.block_size[nb] = block_size;

        node_layer.block_size[nblocks - 1] = (size - (nblocks - 1) * block_size);

        uint64_t cntr = 0;
        node_layer.block_indx[0] = cntr;
        for (uint32_t nb = 0; nb < nblocks; ++nb) {
            cntr += node_layer.block_size[nb];
            node_layer.block_indx[nb + 1] = cntr;
        }
    }

    void get_part(const uint64_t &block_size) { // restores partition for already distributed matrix
        uint64_t cntr;
        uint64_t block_size_ = block_size;

        core_layer.nblocks = id.nm_ncores;
        core_layer.block_size.resize(core_layer.nblocks);
        core_layer.block_indx.resize(core_layer.nblocks + 1);
        // mpi::allgather<uint64_t>(&block_size_, 1, core_layer.block_size.data(), 1, mpi::INTRA_NUMA);

        uint64_t numa_size;
        mpi::allreduce_sum<uint64_t>(&block_size_, &numa_size, 1, mpi::INTRA_NUMA);
        uint64_t core_size = misc::split_range<float64_t>(numa_size, id.nm_ncores);

        uint64_t core_offset = std::min(numa_size, core_size * id.nm_core);
        core_size = std::min(core_size, numa_size - core_offset);
        mpi::allgather<uint64_t>(&core_size, 1, core_layer.block_size.data(), 1, mpi::INTRA_NUMA);

        core_layer.block_indx[0] = 0;
        for (uint32_t nb = 0; nb < core_layer.nblocks; ++nb)
            core_layer.block_indx[nb + 1] = core_layer.block_indx[nb] + core_layer.block_size[nb];
        ////
        numa_layer.nblocks = id.nd_nnumas;
        numa_layer.block_size.resize(numa_layer.nblocks);
        numa_layer.block_indx.resize(numa_layer.nblocks + 1);
        mpi::allreduce_sum<uint64_t>(&block_size_, &(numa_layer.block_size[id.nd_numa]), 1,
                                     mpi::INTRA_NUMA);

        if (id.numa_master_process()) {
            uint64_t temp = numa_layer.block_size[id.nd_numa];
            mpi::allgather<uint64_t>(&temp, 1, numa_layer.block_size.data(), 1, mpi::CROSS_NUMA);
        }
        mpi::bcast<uint64_t>(numa_layer.block_size.data(), numa_layer.nblocks, 0, mpi::INTRA_NUMA);

        numa_layer.block_indx[0] = 0;
        for (uint32_t nb = 0; nb < numa_layer.nblocks; ++nb)
            numa_layer.block_indx[nb + 1] = numa_layer.block_indx[nb] + numa_layer.block_size[nb];

        node_layer.nblocks = id.gl_nnodes;
        node_layer.block_size.resize(node_layer.nblocks);
        node_layer.block_indx.resize(node_layer.nblocks + 1);
        mpi::allreduce_sum<uint64_t>(&block_size_, &(node_layer.block_size[id.gl_node]), 1,
                                     mpi::INTRA_NODE);

        if (id.node_master_process()) {
            uint64_t temp = node_layer.block_size[id.gl_node];
            mpi::allgather<uint64_t>(&temp, 1, node_layer.block_size.data(), 1, mpi::CROSS_NODE);
        }
        mpi::bcast<uint64_t>(node_layer.block_size.data(), node_layer.nblocks, 0, mpi::INTRA_NODE);

        node_layer.block_indx[0] = 0;
        for (uint32_t nb = 0; nb < node_layer.nblocks; ++nb)
            node_layer.block_indx[nb + 1] = node_layer.block_indx[nb] + node_layer.block_size[nb];
    }
};

static inline std::shared_ptr<part> get_shared_part() {
    return std::shared_ptr<part>(new part);
}

static inline std::shared_ptr<part> make_partitioner(uint64_t local_size) {
    auto p = std::shared_ptr<part>(new part);
    p->get_part(local_size);
    return p;
}

} // namespace part
} // namespace XAMG
