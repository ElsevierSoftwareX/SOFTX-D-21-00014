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
void split_by_rows(const csr_matrix<F, I1, I2, I3, I4> &mat,
                   std::vector<csr_matrix<F, I1, I2, I3, I4>> &blocks,
                   const part::part_layer &layer) {
    assert(mat.sharing_mode == mem::CORE);
    uint32_t nblocks = layer.nblocks;
    blocks.resize(nblocks);
    if (mat.if_empty)
        return;

    uint32_t r1, r2;
    std::vector<F> val_;
    std::vector<uint64_t> col_;

    r1 = r2 = 0;
    for (uint32_t nb = 0; nb < nblocks; ++nb) {
        blocks[nb].ncols = mat.ncols;
        blocks[nb].block_nrows = layer.block_size[nb];
        blocks[nb].block_ncols = mat.block_ncols;
        blocks[nb].block_row_offset = mat.block_row_offset + layer.block_indx[nb];
        blocks[nb].block_col_offset = mat.block_col_offset;

        blocks[nb].if_indexed = mat.if_indexed;

        if (!mat.if_indexed) {
            // r1 = layer.block_indx[nb];
            r2 = r1 + layer.block_size[nb];
            blocks[nb].nrows = blocks[nb].block_nrows;
        } else {
            mat.row_ind.check(vector::vector::initialized);

            r2 = mat.nrows;
            const auto &row_ind_ptr = mat.row_ind.template get_aligned_ptr<I3>();
            for (uint32_t i = r1; i < mat.nrows; ++i) {
                if (row_ind_ptr[i] >= layer.block_indx[nb] + layer.block_size[nb]) {
                    r2 = i;
                    break;
                }
            }
            blocks[nb].nrows = r2 - r1;
        }

        blocks[nb].nonzeros = mat.get_range_size(r1, r2);
        blocks[nb].alloc();

        if (!blocks[nb].if_empty) {
            for (uint64_t i = r1; i < r2; ++i) {
                uint64_t ii = i - r1;

                mat.get_row(i, col_, val_);
                blocks[nb].upload_row(ii, col_, val_, col_.size());
            }

            if (mat.if_indexed) {
                blocks[nb].col_ind = mat.col_ind;
                const auto row_ind_ptr = mat.row_ind.template get_aligned_ptr<I3>();
                auto block_row_ind_ptr = blocks[nb].row_ind.template get_aligned_ptr<I3>();

                for (uint64_t i = r1; i < r2; ++i) {
                    uint64_t ii = i - r1;
                    block_row_ind_ptr[ii] = row_ind_ptr[i] - layer.block_indx[nb];
                }

                blocks[nb].row_ind.if_zero = false;
                blocks[nb].row_ind.if_initialized = true;
            }

            blocks[nb].row.if_zero = false;
            blocks[nb].row.if_initialized = true;
        }
        r1 = r2;
    }
    // for (uint32_t nb = 0; nb < nblocks; ++nb)
    //     blocks[nb].print();
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void split_by_columns(const csr_matrix<F, I1, I2, I3, I4> &mat,
                      std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks,
                      const part::part_layer &layer) {
    if (mat.if_empty)
        return;

    uint32_t nblocks = layer.nblocks;
    const std::vector<uint64_t> &col_block_size = layer.block_size;
    const std::vector<uint64_t> &col_block_indx = layer.block_indx;

    /////////

    // vector contains mapping of the column index to the specific block (block index is unmapped)
    std::vector<uint32_t> col_to_block(mat.nonzeros, 0);
    std::vector<uint64_t> nnz_per_block(nblocks, 0);

    mat.row.check(vector::vector::initialized);
    mat.col.check(vector::vector::initialized);
    mat.val.check(vector::vector::initialized);
    auto mat_row_ptr = mat.row.template get_aligned_ptr<I1>();
    auto mat_col_ptr = mat.col.template get_aligned_ptr<I2>();
    auto mat_val_ptr = mat.val.template get_aligned_ptr<F>();
    assert(mat_row_ptr[mat.nrows] == mat.nonzeros);

    for (uint64_t j = 0; j < mat.nonzeros; ++j) {
        col_to_block[j] = XAMG::misc::search_range(mat_col_ptr[j], col_block_indx);
        nnz_per_block[col_to_block[j]]++;
    }

    /////////

    std::vector<uint64_t> block_map_proc_to_id(nblocks, 0);

    uint32_t non_empty_blocks = 0;
    for (uint64_t i = 0; i < nnz_per_block.size(); ++i) {
        if (nnz_per_block[i]) {
            block_map_proc_to_id[i] = non_empty_blocks;
            ++non_empty_blocks;
        }
    }

    for (uint64_t i = 0; i < col_to_block.size(); ++i)
        col_to_block[i] = block_map_proc_to_id[col_to_block[i]];

    /////////

    assert(csr_chunks.size() == 0);
    csr_chunks.resize(non_empty_blocks);

    uint32_t nb = 0;
    for (uint64_t rnb = 0; rnb < nnz_per_block.size(); ++rnb) {
        if (nnz_per_block[rnb]) {
            csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[nb]);

            chunk_wrapper.proc = rnb;

            chunk_wrapper.mtx.nrows = mat.nrows;
            chunk_wrapper.mtx.ncols = col_block_size[rnb];
            chunk_wrapper.mtx.block_nrows = chunk_wrapper.mtx.nrows;
            chunk_wrapper.mtx.block_ncols = chunk_wrapper.mtx.ncols;

            chunk_wrapper.mtx.block_row_offset = mat.block_row_offset;
            chunk_wrapper.mtx.block_col_offset = col_block_indx[rnb];

            chunk_wrapper.mtx.nonzeros = nnz_per_block[rnb];

            chunk_wrapper.mtx.alloc();
            ++nb;
        }
    }

    // XAMG::out << "Block splitting completed...\n";
    // XAMG::io::sync();

    /////////

    uint64_t indx;

    std::vector<uint32_t *> blocks_row_b(non_empty_blocks);
    std::vector<uint32_t *> blocks_col_b(non_empty_blocks);
    std::vector<F *> blocks_val_b(non_empty_blocks);

    for (nb = 0; nb < non_empty_blocks; ++nb) {
        csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[nb]);

        blocks_row_b[nb] = chunk_wrapper.mtx.row.template get_aligned_ptr<uint32_t>();
        blocks_col_b[nb] = chunk_wrapper.mtx.col.template get_aligned_ptr<uint32_t>();
        blocks_val_b[nb] = chunk_wrapper.mtx.val.template get_aligned_ptr<F>();

        chunk_wrapper.mtx.row.if_initialized = true;
        chunk_wrapper.mtx.col.if_initialized = true;
        chunk_wrapper.mtx.val.if_initialized = true;

        chunk_wrapper.mtx.row.if_zero = false;
        chunk_wrapper.mtx.col.if_zero = false;
        chunk_wrapper.mtx.val.if_zero = false;
    }

    for (nb = 0; nb < non_empty_blocks; ++nb)
        blocks_row_b[nb][0] = 0;

    for (uint64_t i = 0; i < mat.nrows; ++i) {
        for (nb = 0; nb < non_empty_blocks; ++nb)
            blocks_row_b[nb][i + 1] = blocks_row_b[nb][i];

        for (uint64_t j = mat_row_ptr[i]; j < mat_row_ptr[i + 1]; ++j) {
            nb = col_to_block[j];
            indx = blocks_row_b[nb][i + 1];
            csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[nb]);

            blocks_col_b[nb][indx] = mat_col_ptr[j] - chunk_wrapper.mtx.block_col_offset;
            blocks_val_b[nb][indx] = mat_val_ptr[j];

            ++blocks_row_b[nb][i + 1];
        }
    }

    for (nb = 0; nb < non_empty_blocks; ++nb) {
        csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[nb]);
        chunk_wrapper.mtx.block_col_offset += mat.block_col_offset;
    }
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void merge_col_blocks(const std::vector<csr_matrix<F, I1, I2, I3, I4>> &blocks,
                      csr_matrix<F, I1, I2, I3, I4> &mat) {
    uint64_t block_nrows_ = 0;
    uint64_t block_nnzs_ = 0;
    for (auto &block : blocks) {
        assert(block.if_indexed == false);

        block_nrows_ += block.get_nrows();
        block_nnzs_ += block.get_nonzeros();
    }

    for (uint32_t nb = 1; nb < blocks.size(); ++nb) {
        assert(blocks[nb - 1].block_row_offset + blocks[nb - 1].get_nrows() ==
               blocks[nb].block_row_offset);
        //    XAMG::out << ALLRANKS << "block size : " << block_nrows_ << " nonzeros : " << block_nnzs_ << std::endl;
    }

    ////

    mat.nrows = block_nrows_;
    mat.ncols = blocks[0].get_ncols();
    mat.nonzeros = block_nnzs_;
    mat.alloc();

    mat.block_nrows = mat.nrows;
    mat.block_ncols = mat.ncols;
    mat.block_row_offset = blocks[0].get_block_row_offset();
    mat.block_col_offset = 0;

    if (mat.if_empty)
        return;

    std::vector<F> val_;
    std::vector<uint64_t> col_;

    for (auto &block : blocks) {
        if (!block.if_empty) {
            uint64_t nnz_check = 0;
            for (uint64_t i = 0; i < block.get_nrows(); ++i) {
                block.get_row(i, col_, val_);
                nnz_check += col_.size();
            }
            // XAMG::out <<ALLRANKS << " " << nnz_check << " " << block.get_nonzeros() << std::endl;
            assert(nnz_check == block.get_nonzeros());
        }
    }

    for (auto &block : blocks) {
        if (!block.if_empty) {
            uint64_t offset = block.get_block_row_offset();

            for (uint64_t i = 0; i < block.get_nrows(); ++i) {
                uint64_t ii = i + (offset - mat.block_row_offset);

                block.get_row(i, col_, val_);
                mat.upload_row(ii, col_, val_, col_.size());
            }
        } else {
            col_.resize(0);
            val_.resize(0);
            uint64_t offset = block.get_block_row_offset();

            for (uint64_t i = 0; i < block.get_nrows(); ++i) {
                uint64_t ii = i + (offset - mat.block_row_offset);
                mat.upload_row(ii, col_, val_, col_.size());
            }
        }
    }
    // merged_csr.print();
}

} // namespace matrix
} // namespace XAMG
