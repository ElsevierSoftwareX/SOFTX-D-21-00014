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
namespace vector {

/*
template <typename F>
void collect(const vector &vec_in, vector &vec_out, mpi::scope comm) {
    int proc, nprocs;
    mpi::assign_proc_info(comm, proc, nprocs);
    int root = 0;

    comm::data_exchange_buffer comm_send_buffer(root);
    std::vector<comm::data_exchange_buffer> comm_recv_buffer;
    vec_in.push_to_buffer<F>(comm_send_buffer);

    buffer_gather(comm_send_buffer, comm_recv_buffer, root, comm);

    if (!proc) {
        std::vector<vector> vec_remote(comm_recv_buffer.size(), mem::CORE);

        for (uint32_t np = 0; np < comm_recv_buffer.size(); ++np)
            vec_remote[np].pull_from_buffer<F>(comm_recv_buffer[np]);

        merge_col_blocks<F>(vec_remote, vec_out);
//        if (!vec_out.if_empty) {
////            XAMG::out << ALLRANKS << row.get_aligned_ptr<I1>()[nrows] << " " << nonzeros <<
std::endl;
//            assert(mat_out.row.template get_aligned_ptr<I1>()[mat_out.nrows] == mat_out.nonzeros);
//        }
    }
}
*/

template <typename F>
void collect_core(const vector &vec_in, std::vector<vector> &vec_remote, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);
    int root = 0;

    comm::data_exchange_buffer comm_send_buffer(root);
    std::vector<comm::data_exchange_buffer> comm_recv_buffer;
    vec_in.push_to_buffer<F>(comm_send_buffer);

    buffer_gather(comm_send_buffer, comm_recv_buffer, root, comm_group.comm);

    if (comm_group.master_proc) {
        vec_remote.resize(comm_recv_buffer.size());

        for (uint32_t np = 0; np < comm_recv_buffer.size(); ++np)
            vec_remote[np].pull_from_buffer<F>(comm_recv_buffer[np]);
    }
}

template <typename F>
void collect(const vector &vec_in, vector &vec_out, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    std::vector<vector> vec_remote;
    collect_core<F>(vec_in, vec_remote, comm_group.comm);

    if (comm_group.master_proc) {
        merge_col_blocks<F>(vec_remote, vec_out);
    }
}

template <typename F>
void collect(std::shared_ptr<vector> &vec, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    std::vector<vector> vec_remote;
    collect_core<F>(*vec, vec_remote, comm_group.comm);

    vec.reset(new vector);
    if (comm_group.master_proc) {
        merge_col_blocks<F>(vec_remote, *vec);
    }
}

/*
template<typename F>
void distribute(const vector &vec_in, vector &vec_out, const part::part_layer &layer, mpi::scope
comm) {
    int proc, nprocs;
    mpi::assign_proc_info(comm, proc, nprocs);
//    if (!proc)
//        assert(!vec_in.if_empty);

    std::vector<comm::data_exchange_buffer> comm_send_buffer;
    comm::data_exchange_buffer comm_recv_buffer;

    std::vector<vector> blocks;

    if (!proc) {
        split_by_rows<F>(vec_in, blocks, layer);
        assert(blocks.size() == (uint32_t)nprocs);

        comm_send_buffer.resize(blocks.size());
        for (uint32_t np = 0; np < blocks.size(); ++np)
            blocks[np].push_to_buffer<F>(comm_send_buffer[np]);
    }

    buffer_scatter(comm_send_buffer, comm_recv_buffer, 0, comm);
    vec_out.pull_from_buffer<F>(comm_recv_buffer);
}
*/

template <typename F>
void distribute_core(const vector &vec_in, comm::data_exchange_buffer &comm_recv_buffer,
                     const part::part_layer &layer, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    std::vector<comm::data_exchange_buffer> comm_send_buffer;
    std::vector<vector> blocks;

    if (comm_group.master_proc) {
        split_by_rows<F>(vec_in, blocks, layer);
        assert(blocks.size() == (uint32_t)comm_group.nprocs);

        comm_send_buffer.resize(blocks.size());
        for (uint32_t np = 0; np < blocks.size(); ++np)
            blocks[np].push_to_buffer<F>(comm_send_buffer[np]);
    }

    buffer_scatter(comm_send_buffer, comm_recv_buffer, 0, comm_group.comm);
}

template <typename F>
void distribute(const vector &vec_in, vector &vec_out, const part::part_layer &layer,
                mpi::scope comm) {
    comm::data_exchange_buffer comm_recv_buffer;
    distribute_core<F>(vec_in, comm_recv_buffer, layer, comm);

    vec_out.pull_from_buffer<F>(comm_recv_buffer);
}

template <typename F>
void distribute(std::shared_ptr<vector> &vec, const part::part_layer &layer, mpi::scope comm) {
    comm::data_exchange_buffer comm_recv_buffer;
    distribute_core<F>(*vec, comm_recv_buffer, layer, comm);

    vec.reset(new vector);
    vec->pull_from_buffer<F>(comm_recv_buffer);
}

template <typename F>
void redistribute(std::shared_ptr<vector> &vec, const map_info &mapping_info,
                  const segment::hierarchy &layer) {
    mpi::comm_pool comm_group(mpi::cross_layer_comm(layer));

    std::vector<vector> vec_send_block(mapping_info.nneighbours);

    for (uint32_t i = 0; i < mapping_info.nneighbours; ++i) {
        vec_send_block[i].alloc<F>(mapping_info.nrows_per_block[i], vec->nv);
        vec_send_block[i].ext_offset = 0;
    }

    uint64_t l;
    std::vector<uint32_t> block_elem_cntr(mapping_info.nneighbours, 0);

    for (uint32_t i = 0; i < mapping_info.mapped_block.size(); ++i) {
        l = block_elem_cntr[mapping_info.mapped_block[i]];

        vec_send_block[mapping_info.mapped_block[i]].set_element<F>(l, vec->get_element<F>(i));

        ++block_elem_cntr[mapping_info.mapped_block[i]];
    }

    for (uint32_t i = 0; i < mapping_info.nneighbours; ++i) {
        vec_send_block[i].if_zero = vec->if_zero;
    }

    ////

    std::vector<comm::data_exchange_buffer> comm_send_buffer;
    std::vector<comm::data_exchange_buffer> comm_recv_buffer;

    comm_send_buffer.reserve(mapping_info.nneighbours);
    for (uint32_t i = 0; i < mapping_info.nneighbours; ++i) {
        comm_send_buffer.emplace_back(mapping_info.list_of_neighbours[i]);
        vec_send_block[i].push_to_buffer<F>(comm_send_buffer.back());
    }

    comm::buffer_alltoall(comm_send_buffer, comm_recv_buffer, comm_group.comm);

    ////////////////////////////////////

    uint64_t proc_nrows = 0;
    uint64_t block_offset = vec->ext_offset;
    mpi::bcast<uint64_t>(&block_offset, 1, 0, comm_group.comm);

    std::vector<vector> vec_recv_block(comm_recv_buffer.size());

    for (size_t i = 0; i < comm_recv_buffer.size(); ++i) {
        vec_recv_block[i].pull_from_buffer<F>(comm_recv_buffer[i]);

        proc_nrows += vec_recv_block[i].size;
    }

    mpi::token tok;
    uint64_t proc_offset = 0; // offset over procs inside this layer block
    if (!comm_group.master_proc) {
        mpi::recv<uint64_t>(&proc_offset, 1, comm_group.proc - 1, comm_group.comm, 0);
    }

    if (comm_group.proc < comm_group.nprocs - 1) {
        uint64_t next_offset = proc_offset + proc_nrows;
        mpi::isend<uint64_t>(&next_offset, 1, comm_group.proc + 1, tok, comm_group.comm, 0);
        mpi::wait(tok);
    }

    ////

    vec.reset(new vector);

    merge_col_blocks<F>(vec_recv_block, *vec);
    vec->ext_offset = proc_offset + block_offset;
}

} // namespace vector
} // namespace XAMG
