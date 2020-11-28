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

template <typename F, typename I1, typename I2, typename I3, typename I4>
void transpose(const csr_matrix<F, I1, I2, I3, I4> &mat, csr_matrix<F, I1, I2, I3, I4> &matT) {
    std::vector<uint32_t> nelems_per_col(mat.ncols, 0);
    std::vector<uint64_t> col_;
    std::vector<float64_t> val_;

    for (uint64_t l = 0; l < mat.nrows; ++l) {
        mat.get_row(l, col_, val_);

        for (uint32_t ii = 0; ii < col_.size(); ++ii)
            ++nelems_per_col[col_[ii]];
    }

    /////////

    matT.nrows = mat.get_ncols();
    matT.ncols = mat.get_nrows();
    matT.nonzeros = mat.get_nonzeros();

    matT.if_indexed = mat.indexed();

    matT.block_nrows = mat.block_ncols;
    matT.block_ncols = mat.block_nrows;

    matT.alloc();
    if (matT.if_empty)
        return;

    matT.row.check(vector::vector::allocated);
    matT.col.check(vector::vector::allocated);
    matT.val.check(vector::vector::allocated);

    auto rowT_ptr = matT.row.template get_aligned_ptr<uint32_t>();
    auto colT_ptr = matT.col.template get_aligned_ptr<uint32_t>();
    auto valT_ptr = matT.val.template get_aligned_ptr<F>();

    rowT_ptr[0] = 0;
    for (uint64_t l = 0; l < matT.nrows; ++l) {
        rowT_ptr[l + 1] = rowT_ptr[l] + nelems_per_col[l];
        nelems_per_col[l] = 0;
    }

    for (uint64_t l = 0; l < mat.nrows; ++l) {
        mat.get_row(l, col_, val_);

        for (uint32_t ii = 0; ii < col_.size(); ++ii) {
            uint32_t rowT = col_[ii];
            uint32_t rowT_offset = rowT_ptr[rowT];

            colT_ptr[rowT_offset + nelems_per_col[rowT]] = l;
            valT_ptr[rowT_offset + nelems_per_col[rowT]] = (F)val_[ii];

            ++nelems_per_col[rowT];
        }
    }
    matT.row.if_initialized = true;
    matT.col.if_initialized = true;
    matT.val.if_initialized = true;

    matT.row.if_zero = false;
    matT.col.if_zero = false;
    matT.val.if_zero = false;

    /////////

    if (matT.if_indexed) {
        auto col_ind_ = mat.get_col_ind_vector();
        col_ind_.check(vector::vector::initialized);
        matT.row.check(vector::vector::allocated);
        matT.row_ind = col_ind_;

        auto row_ind_ = mat.get_row_ind_vector();
        row_ind_.check(vector::vector::initialized);
        matT.col.check(vector::vector::allocated);
        matT.col_ind = row_ind_;
    }
    //    csr_block.print();
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void transpose(const std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_node_chunks,
               std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_nodeT_chunks) {
    std::vector<comm::data_exchange_buffer> comm_send_buffer;
    std::vector<comm::data_exchange_buffer> comm_recv_buffer;

    if (id.node_master_process()) {
        for (uint32_t i = 0; i < csr_node_chunks.size(); ++i) {
            const_csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_node_chunks[i]);
            csr_matrix<F, I1, I2, I3, I4> blockT;
            transpose(chunk_wrapper.mtx, blockT);

            comm_send_buffer.emplace_back(chunk_wrapper.proc);
            blockT.push_to_buffer(comm_send_buffer.back());
        }

        comm::buffer_alltoall(comm_send_buffer, comm_recv_buffer, mpi::CROSS_NODE);

        csr_nodeT_chunks.resize(comm_recv_buffer.size());

        for (uint16_t nb = 0; nb < comm_recv_buffer.size(); ++nb) {
            csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunkT_wrapper(csr_nodeT_chunks[nb]);
            chunkT_wrapper.mtx.pull_from_buffer(comm_recv_buffer[nb]);
            chunkT_wrapper.proc = comm_recv_buffer[nb].nbr;
        }
    }
}

} // namespace matrix
} // namespace XAMG
