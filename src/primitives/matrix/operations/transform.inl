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
namespace matrix {

/*
template<typename F, typename I1, typename I2, typename I3, typename I4>
void distribute(const csr_matrix<F, I1, I2, I3, I4> &mat_in,
        csr_matrix<F, I1, I2, I3, I4> &mat_out,
        const part::part_layer &layer, mpi::scope comm) {
    int proc, nprocs;
    mpi::assign_proc_info(comm, proc, nprocs);
    if (!proc) {
        assert(!mat_in.if_empty);
    }

    std::vector<comm::data_exchange_buffer> comm_send_buffer;
    comm::data_exchange_buffer comm_recv_buffer;

    std::vector<csr_matrix<F, I1, I2, I3, I4>> blocks;

    if (!proc) {
        split_by_rows(mat_in, blocks, layer);
        assert(blocks.size() == (uint32_t)nprocs);

        comm_send_buffer.resize(blocks.size());
        for (uint32_t np = 0; np < blocks.size(); ++np)
            blocks[np].push_to_buffer(comm_send_buffer[np]);
    }

    buffer_scatter(comm_send_buffer, comm_recv_buffer, 0, comm);
    mat_out.pull_from_buffer(comm_recv_buffer);
}
*/

template <typename F, typename I1, typename I2, typename I3, typename I4>
void distribute_core(const csr_matrix<F, I1, I2, I3, I4> &mat_in,
                     comm::data_exchange_buffer &comm_recv_buffer, const part::part_layer &layer,
                     mpi::scope comm) {
    mpi::comm_pool comm_group(comm);
    if (comm_group.master_proc)
        assert(!mat_in.if_empty);

    std::vector<comm::data_exchange_buffer> comm_send_buffer;
    std::vector<csr_matrix<F, I1, I2, I3, I4>> blocks;

    if (comm_group.master_proc) {
        split_by_rows(mat_in, blocks, layer);
        assert(blocks.size() == (uint32_t)comm_group.nprocs);

        comm_send_buffer.resize(blocks.size());
        for (uint32_t np = 0; np < blocks.size(); ++np)
            blocks[np].push_to_buffer(comm_send_buffer[np]);
    }

    buffer_scatter(comm_send_buffer, comm_recv_buffer, 0, comm_group.comm);
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void distribute(const csr_matrix<F, I1, I2, I3, I4> &mat_in, csr_matrix<F, I1, I2, I3, I4> &mat_out,
                const part::part_layer &layer, mpi::scope comm) {
    comm::data_exchange_buffer comm_recv_buffer;
    distribute_core(mat_in, comm_recv_buffer, layer, comm);

    mat_out.pull_from_buffer(comm_recv_buffer);
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void distribute(std::shared_ptr<csr_matrix<F, I1, I2, I3, I4>> &mat, const part::part_layer &layer,
                mpi::scope comm) {
    comm::data_exchange_buffer comm_recv_buffer;
    distribute_core(*mat, comm_recv_buffer, layer, comm);

    mat.reset(new XAMG::matrix::csr_matrix<F, I1, I2, I3, I4>);
    mat->pull_from_buffer(comm_recv_buffer);
}

template <typename F>
void distribute(const dense_matrix<F> &mat_in, dense_matrix<F> &mat_out,
                const part::part_layer &layer, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    uint8_t empty_flag = mat_in.if_empty;
    mpi::bcast<uint8_t>(&empty_flag, 1, 0, comm_group.comm);
    if (empty_flag)
        return;

    std::vector<comm::data_exchange_buffer> comm_send_buffer;
    comm::data_exchange_buffer comm_recv_buffer;

    std::vector<dense_matrix<F>> blocks;

    if (comm_group.master_proc) {
        split_by_rows(mat_in, blocks, layer);
        assert(blocks.size() == (uint32_t)comm_group.nprocs);

        comm_send_buffer.resize(blocks.size());
        for (uint32_t np = 0; np < blocks.size(); ++np)
            blocks[np].push_to_buffer(comm_send_buffer[np]);
    }

    buffer_scatter(comm_send_buffer, comm_recv_buffer, 0, comm_group.comm);
    mat_out.pull_from_buffer(comm_recv_buffer);
}

/////////
/*
template<typename F, typename I1, typename I2, typename I3, typename I4>
void collect(const csr_matrix<F, I1, I2, I3, I4> &mat_in,
        csr_matrix<F, I1, I2, I3, I4> &mat_out, mpi::scope comm) {
    int proc, nprocs;
    mpi::assign_proc_info(comm, proc, nprocs);
    int root = 0;

    comm::data_exchange_buffer comm_send_buffer(root);
    std::vector<comm::data_exchange_buffer> comm_recv_buffer;
    mat_in.push_to_buffer(comm_send_buffer);
//    comm_send_buffer.nbr = root;

    buffer_gather(comm_send_buffer, comm_recv_buffer, root, comm);

    if (!proc) {
        std::vector<csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t>>
mat_remote(comm_recv_buffer.size(), mem::CORE);

        for (uint32_t np = 0; np < comm_recv_buffer.size(); ++np)
            mat_remote[np].pull_from_buffer(comm_recv_buffer[np]);

        merge_col_blocks(mat_remote, mat_out);
        if (!mat_out.if_empty) {
//            XAMG::out << ALLRANKS << row.get_aligned_ptr<I1>()[nrows] << " " << nonzeros <<
std::endl; assert(mat_out.row.template get_aligned_ptr<I1>()[mat_out.nrows] == mat_out.nonzeros);
        }
    }
}
*/

template <typename F, typename I1, typename I2, typename I3, typename I4>
void collect_core(const csr_matrix<F, I1, I2, I3, I4> &mat_in,
                  std::vector<csr_matrix<F, I1, I2, I3, I4>> &mat_remote, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);
    int root = 0;

    comm::data_exchange_buffer comm_send_buffer(root);
    std::vector<comm::data_exchange_buffer> comm_recv_buffer;
    mat_in.push_to_buffer(comm_send_buffer);

    buffer_gather(comm_send_buffer, comm_recv_buffer, root, comm_group.comm);

    if (comm_group.master_proc) {
        mat_remote.resize(comm_recv_buffer.size());
        for (uint32_t np = 0; np < comm_recv_buffer.size(); ++np)
            mat_remote[np].pull_from_buffer(comm_recv_buffer[np]);
    }
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void collect(const csr_matrix<F, I1, I2, I3, I4> &mat_in, csr_matrix<F, I1, I2, I3, I4> &mat_out,
             mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    std::vector<csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t>> mat_remote;
    collect_core(mat_in, mat_remote, comm);

    if (comm_group.master_proc) {
        merge_col_blocks(mat_remote, mat_out);
        if (!mat_out.if_empty) {
            // XAMG::out << ALLRANKS << row.get_aligned_ptr<I1>()[nrows] << " " << nonzeros << std::endl;
            assert(mat_out.row.template get_aligned_ptr<I1>()[mat_out.nrows] == mat_out.nonzeros);
        }
    }
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void collect(std::shared_ptr<csr_matrix<F, I1, I2, I3, I4>> &mat, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    std::vector<csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t>> mat_remote;
    collect_core(*mat, mat_remote, comm_group.comm);

    mat.reset(new XAMG::matrix::csr_matrix<F, I1, I2, I3, I4>);
    if (comm_group.master_proc) {
        merge_col_blocks(mat_remote, *mat);
        if (!mat->if_empty) {
            // XAMG::out << ALLRANKS << row.get_aligned_ptr<I1>()[nrows] << " " << nonzeros << std::endl;
            assert(mat->row.template get_aligned_ptr<I1>()[mat->nrows] == mat->nonzeros);
        }
    }
}

template <typename F>
void collect(const dense_matrix<F> &mat_in, dense_matrix<F> &mat_out, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);
    int root = 0;

    comm::data_exchange_buffer comm_send_buffer;
    std::vector<comm::data_exchange_buffer> comm_recv_buffer;
    mat_in.push_to_buffer(comm_send_buffer);
    comm_send_buffer.nbr = root;

    buffer_gather(comm_send_buffer, comm_recv_buffer, root, comm_group.comm);

    if (comm_group.master_proc) {
        std::vector<dense_matrix<F>> mat_remote(comm_recv_buffer.size(), mem::CORE);

        for (uint32_t np = 0; np < comm_recv_buffer.size(); ++np)
            mat_remote[np].pull_from_buffer(comm_recv_buffer[np]);

        merge_col_blocks(mat_remote, mat_out);
        // mat_out.print();
    }
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void redistribute(std::shared_ptr<csr_matrix<F, I1, I2, I3, I4>> &mat, const map_info &mapping_info,
                  const segment::hierarchy &layer) {
    mpi::comm_pool comm_group(mpi::cross_layer_comm(layer));

    std::vector<uint32_t> nonzeros_per_block(mapping_info.nneighbours, 0);
    for (uint32_t i = 0; i < mapping_info.mapped_block.size(); ++i) {
        nonzeros_per_block[mapping_info.mapped_block[i]] += mat->get_row_size(i);
    }

    std::vector<XAMG::matrix::csr_matrix<F, I1, I2, I3, I4>> csr_send_block(
        mapping_info.nneighbours);

    for (uint32_t i = 0; i < mapping_info.nneighbours; ++i) {
        csr_send_block[i].nrows = mapping_info.nrows_per_block[i];
        csr_send_block[i].nonzeros = nonzeros_per_block[i];
        csr_send_block[i].ncols = mat->ncols;
        csr_send_block[i].block_nrows = csr_send_block[i].nrows;
        csr_send_block[i].block_ncols = csr_send_block[i].ncols;
        csr_send_block[i].block_row_offset = 0;
        csr_send_block[i].block_col_offset = 0;
        csr_send_block[i].alloc();
    }

    std::vector<uint64_t> _col;
    std::vector<float64_t> _val;
    std::vector<F> _val_F;

    std::vector<uint32_t> block_row_cntr(mapping_info.nneighbours, 0);

    for (uint32_t i = 0; i < mapping_info.mapped_block.size(); ++i) {
        mat->get_row(i, _col, _val);

        uint64_t l = block_row_cntr[mapping_info.mapped_block[i]];

        _val_F.resize(_val.size());
        for (size_t ii = 0; ii < _val.size(); ++ii)
            _val_F[ii] = (F)_val[ii];
        csr_send_block[mapping_info.mapped_block[i]].upload_row(l, _col, _val_F, _col.size());
        ++block_row_cntr[mapping_info.mapped_block[i]];
    }

    ////

    std::vector<comm::data_exchange_buffer> comm_send_buffer;
    std::vector<comm::data_exchange_buffer> comm_recv_buffer;

    comm_send_buffer.reserve(mapping_info.nneighbours);
    for (uint32_t i = 0; i < mapping_info.nneighbours; ++i) {
        comm_send_buffer.emplace_back(mapping_info.list_of_neighbours[i]);
        csr_send_block[i].push_to_buffer(comm_send_buffer.back());
    }

    comm::buffer_alltoall(comm_send_buffer, comm_recv_buffer, comm_group.comm);

    ////////////////////////////////////

    uint64_t proc_nrows = 0;
    uint64_t block_offset = mat->block_row_offset;
    mpi::bcast<uint64_t>(&block_offset, 1, 0, comm_group.comm);

    std::vector<csr_matrix<F, I1, I2, I3, I4>> csr_recv_block(comm_recv_buffer.size());
    for (size_t i = 0; i < comm_recv_buffer.size(); ++i) {
        csr_recv_block[i].pull_from_buffer(comm_recv_buffer[i]);

        proc_nrows += csr_recv_block[i].get_nrows();
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

    proc_nrows = 0;
    for (auto &csr_block : csr_recv_block) {
        csr_block.block_row_offset = proc_offset + block_offset + proc_nrows;
        proc_nrows += csr_block.nrows;
    }

    mat.reset(new csr_matrix<F, I1, I2, I3, I4>);
    merge_col_blocks(csr_recv_block, *mat);
}

} // namespace matrix
} // namespace XAMG
