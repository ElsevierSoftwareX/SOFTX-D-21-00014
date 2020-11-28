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

template <typename F>
void split_by_rows(const dense_matrix<F> &mat, std::vector<dense_matrix<F>> &blocks,
                   const part::part_layer &layer) {
    // assert(alloc_mode == mem::LOCAL);
    assert(mat.if_indexed == false);

    uint32_t nblocks = layer.nblocks;
    blocks.resize(nblocks, dense_matrix<F>(mem::CORE));
    if (mat.if_empty)
        return;

    uint32_t r1, r2;
    std::vector<float64_t> val_;
    std::vector<uint64_t> col_;

    r1 = r2 = 0;
    for (uint32_t nb = 0; nb < nblocks; ++nb) {
        blocks[nb].ncols = mat.ncols;
        blocks[nb].block_nrows = layer.block_size[nb];
        blocks[nb].block_ncols = mat.block_ncols;
        blocks[nb].block_row_offset = mat.block_row_offset + layer.block_indx[nb];
        blocks[nb].block_col_offset = mat.block_col_offset;

        blocks[nb].if_indexed = mat.if_indexed;

        assert(!mat.if_indexed);
        r2 = r1 + layer.block_size[nb];
        blocks[nb].nrows = blocks[nb].block_nrows;

        assert(blocks[nb].nrows * blocks[nb].ncols == mat.get_range_size(r1, r2));
        blocks[nb].alloc();

        if (!blocks[nb].if_empty) {
            for (uint64_t i = r1; i < r2; ++i) {
                uint64_t ii = i - r1;

                mat.get_row(i, col_, val_);
                blocks[nb].upload_row(ii, val_, val_.size());
            }

            blocks[nb].val.if_zero = false;
            blocks[nb].val.if_initialized = true;
        }
        r1 = r2;
    }
}

template <typename F>
void merge_row_blocks(const std::vector<dense_matrix<F>> &blocks,
                      dense_matrix<F> &mat) { // merge row of blocks
    bool non_empty_block = false;
    uint32_t first_block = 0;
    for (uint32_t nb = 0; nb < blocks.size(); ++nb) {
        if (!blocks[nb].if_empty) {
            if (!non_empty_block)
                first_block = nb;
            non_empty_block = true;
            // blocks[nb].val.check(vector::vector::initialized);
        }
    }
    if (!non_empty_block)
        return;

    uint64_t min_row_range = blocks[first_block].block_row_offset;
    uint64_t max_row_range = blocks[first_block].block_row_offset + blocks[first_block].nrows;
    uint64_t min_col_range = blocks[first_block].block_col_offset;
    uint64_t max_col_range = blocks[first_block].block_col_offset + blocks[first_block].ncols;

    for (uint32_t nb = first_block + 1; nb < blocks.size(); ++nb) {
        if (!blocks[nb].if_empty) {
            min_row_range = std::min(min_row_range, blocks[nb].block_row_offset);
            max_row_range = std::max(max_row_range, blocks[nb].block_row_offset + blocks[nb].nrows);
            min_col_range = std::min(min_col_range, blocks[nb].block_col_offset);
            max_col_range = std::max(max_col_range, blocks[nb].block_col_offset + blocks[nb].ncols);
        }
    }
    mat.block_row_offset = min_row_range;
    mat.nrows = max_row_range - min_row_range;
    mat.block_col_offset = min_col_range;
    mat.ncols = max_col_range - min_col_range;

    mat.block_nrows = mat.nrows;
    mat.block_ncols = mat.ncols;

    for (uint32_t nb = 0; nb < blocks.size(); ++nb) {
        // XAMG::out << ALLRANKS << "block[" << nb << "] : " << blocks[nb].nrows << std::endl;
        assert(blocks[nb].nrows == mat.nrows);
        assert(!blocks[nb].if_indexed);
    }
    // XAMG::out << ALLRANKS << "block_nrows : " << mat.nrows << " row_offset " << mat.block_row_offset << std::endl; XAMG::out << ALLRANKS << "block_ncols : " << mat.ncols << " col_offset " << mat.block_col_offset << std::endl;
    mat.alloc();

    /////////

    mat.val.check(vector::vector::allocated);
    auto val_ptr = mat.val.template get_aligned_ptr<F>();
    blas::set_const<F, 1>(mat.val, 0.0, true);

    for (uint32_t nb = 0; nb < blocks.size(); ++nb) {
        if (!blocks[nb].if_empty) {
            for (uint32_t i = 0; i < blocks[nb].nrows; ++i) {
                uint64_t offset =
                    (i + blocks[nb].get_block_row_offset() - mat.block_row_offset) * mat.ncols +
                    (blocks[nb].get_block_col_offset() - mat.block_col_offset);
                auto val_b_ptr =
                    blocks[nb].val.template get_aligned_ptr<F>() + i * blocks[nb].ncols;

                for (uint64_t j = 0; j < blocks[nb].ncols; ++j)
                    val_ptr[offset + j] = val_b_ptr[j];
            }
        }
    }

    mat.if_empty = false;
    mat.val.if_initialized = true;
    mat.val.if_zero = false;
}

template <typename F>
void merge_col_blocks(const std::vector<dense_matrix<F>> &blocks,
                      dense_matrix<F> &mat) { // merge column of blocks
#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "merge_col_blocks, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
    bool non_empty_block = false;
    uint32_t first_block = 0;
    for (uint32_t nb = 0; nb < blocks.size(); ++nb) {
        if (!blocks[nb].if_empty) {
            if (!non_empty_block)
                first_block = nb;
            non_empty_block = true;
            //        blocks[nb].val.check(vector::vector::initialized);
        }
    }
    if (!non_empty_block)
        return;

    uint64_t min_row_range = blocks[first_block].block_row_offset;
    uint64_t max_row_range = blocks[first_block].block_row_offset + blocks[first_block].nrows;
    uint64_t min_col_range = blocks[first_block].block_col_offset;
    uint64_t max_col_range = blocks[first_block].block_col_offset + blocks[first_block].ncols;

    for (uint32_t nb = first_block + 1; nb < blocks.size(); ++nb) {
        if (!blocks[nb].if_empty) {
            min_row_range = std::min(min_row_range, blocks[nb].block_row_offset);
            max_row_range = std::max(max_row_range, blocks[nb].block_row_offset + blocks[nb].nrows);
            min_col_range = std::min(min_col_range, blocks[nb].block_col_offset);
            max_col_range = std::max(max_col_range, blocks[nb].block_col_offset + blocks[nb].ncols);
        }
    }
    mat.block_row_offset = min_row_range;
    mat.nrows = max_row_range - min_row_range;
    mat.block_col_offset = min_col_range;
    mat.ncols = max_col_range - min_col_range;

    mat.block_nrows = mat.nrows;
    mat.block_ncols = mat.ncols;

    for (uint32_t nb = 0; nb < blocks.size(); ++nb) {
        assert(!blocks[nb].if_indexed);
    }
    mat.alloc();

    /////////

    mat.val.check(vector::vector::allocated);
    auto val_ptr = mat.val.template get_aligned_ptr<F>();
    blas::set_const<F, 1>(mat.val, 0.0, true);

    for (uint32_t nb = 0; nb < blocks.size(); ++nb) {
        if (!blocks[nb].if_empty) {
            for (uint32_t i = 0; i < blocks[nb].nrows; ++i) {
                uint64_t offset =
                    (i + blocks[nb].get_block_row_offset() - mat.block_row_offset) * mat.ncols +
                    (blocks[nb].get_block_col_offset() - mat.block_col_offset);
                auto val_b_ptr =
                    blocks[nb].val.template get_aligned_ptr<F>() + i * blocks[nb].ncols;

                for (uint64_t j = 0; j < blocks[nb].ncols; ++j)
                    val_ptr[offset + j] = val_b_ptr[j];
            }
        }
    }

    mat.if_empty = false;
    mat.val.if_initialized = true;
    mat.val.if_zero = false;
}

//
// template<typename F>
// void matrix::segment(dense_matrix<F> &mat_dense, const uint32_t nblocks) {
////    segment matrix slice by blocks
//    mat_dense.val.check(vector::vector::initialized);
//
//    std::vector<uint64_t> &row_block_size = row_part->node_layer.block_size;
//    std::vector<uint64_t> &row_block_indx = row_part->node_layer.block_indx;
//
//    std::vector<uint64_t> &col_block_size = col_part->node_layer.block_size;
//    std::vector<uint64_t> &col_block_indx = col_part->node_layer.block_indx;
//    assert(col_block_size.size() == nblocks);
//
//////
//
////    XAMG::out << XAMG::ALLRANKS << "Dense:" << std::endl;
////    mat_dense.print();
//
///////////
//
//    std::vector<dense_matrix<F>> blocks;
//    for (uint32_t nb = 0; nb < nblocks; ++nb) {
//        blocks.emplace_back(dense_matrix<F>(mem::LOCAL));
//        blocks[nb].nrows = row_block_size[id.gl_proc];
//        blocks[nb].ncols = col_block_size[nb];
//        blocks[nb].block_nrows = blocks[nb].nrows;
//        blocks[nb].block_ncols = blocks[nb].ncols;
//
//        blocks[nb].block_row_offset = row_block_indx[id.gl_proc];
//        blocks[nb].block_col_offset = col_block_indx[nb];
//
//        blocks[nb].alloc();
//    }
//
///////////
//
//    auto val_ptr = mat_dense.val.template get_aligned_ptr<F>();
//
//    for (uint32_t i = 0; i < mat_dense.nrows; ++i) {
//        auto val_ptr_i = val_ptr + i * mat_dense.ncols;
//
//        for (uint32_t nb = 0; nb < nblocks; ++nb) {
//            blocks[nb].val.check(vector::vector::allocated);
//            auto val_ptr_b_i = blocks[nb].val.template get_aligned_ptr<F>() + i *
//            blocks[nb].ncols;
//
//            uint64_t offset = blocks[nb].block_col_offset;
//            for (uint32_t j = 0; j < blocks[nb].ncols; ++j) {
//                val_ptr_b_i[j] = val_ptr_i[offset + j];
//            }
//        }
//    }
//    for (uint32_t nb = 0; nb < nblocks; ++nb) {
//        blocks[nb].val.if_initialized = true;
//        blocks[nb].val.if_zero = false;
//    }
//
////    XAMG::io::sync();
//
///////////
//
//    std::vector<uint64_t> block_mapping_uncompress; //(1, id.gl_proc);
//    if (nblocks == 1) {
//        block_mapping_uncompress.push_back(id.gl_proc);
//    }
//    else {
//        for (uint32_t nb = 0; nb < (uint32_t)id.gl_nprocs; ++nb) {
//            block_mapping_uncompress.push_back(nb);
//
//            if (nb != (uint32_t)id.gl_proc)
//                blocks[nb].compress();
//        }
//    }
//
//    assemble(blocks, block_mapping_uncompress);
//
///////////
//
//    set_info();
//}

} // namespace matrix
} // namespace XAMG
