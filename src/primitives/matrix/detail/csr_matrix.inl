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
bool csr_matrix<F, I1, I2, I3, I4>::alloc() {
    assert((sharing_mode == mem::CORE) || (sharing_mode == mem::NUMA));
    if (nonzeros) {
        if_empty = false;

        uint64_t row_size = nrows + 1;
        row.alloc<I1>(row_size);

        col.alloc<I2>(nonzeros);
        val.alloc<F>(nonzeros);

        if (if_indexed) {
            row_ind.alloc<I3>(nrows);
            col_ind.alloc<I4>(ncols);

            row_ind.ext_offset = block_row_offset;
            col_ind.ext_offset = block_col_offset;
        }
    }

    return true;
}

// template<typename F, typename I1, typename I2, typename I3, typename I4>
// bool csr_matrix<F, I1, I2, I3, I4>::realloc() {
////    uint64_t row_size = nrows+1;
////    row.alloc<I1>(row_size, mem::LOCAL);
//
////    col.alloc<I2>(nonzeros, mem::LOCAL);
////    val.alloc<F>(nonzeros, mem::LOCAL);
//
////  BK: Hack: imitation of real reallocation
//    col.size = nonzeros;
//    val.size = nonzeros;
//
////    if (if_indexed) {
////        row_ind.alloc<I3>(nrows, mem::LOCAL);
////        col_ind.alloc<I4>(ncols, mem::LOCAL);
////    }
//
//    return true;
//}

static inline void bcast_vector_state(vector::vector &vec, mpi::scope comm) {
    uint8_t i8;
    i8 = vec.if_initialized;
    mpi::bcast<uint8_t>(&i8, 1, 0, comm);
    vec.if_initialized = i8;

    i8 = vec.if_zero;
    mpi::bcast<uint8_t>(&i8, 1, 0, comm);
    vec.if_zero = i8;
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void csr_matrix<F, I1, I2, I3, I4>::compress() {
    if (if_empty)
        return;

    row.check(vector::vector::initialized);
    col.check(vector::vector::initialized);

    uint64_t non_empty_rows = 0;
    auto row_ptr = row.get_aligned_ptr<I1>();
    auto col_ptr = col.get_aligned_ptr<I2>();

    for (uint64_t i = 0; i < nrows; ++i) {
        if (row_ptr[i + 1] - row_ptr[i])
            ++non_empty_rows;
    }

    // row_ind.check(vector::vector::unallocated);
    row_ind.alloc<I3>(non_empty_rows);
    auto row_ind_ptr = row_ind.get_aligned_ptr<I3>();

    non_empty_rows = 0;
    for (uint64_t i = 0; i < nrows; ++i) {
        if (row_ptr[i + 1] - row_ptr[i]) {
            row_ind_ptr[non_empty_rows] = i;
            ++non_empty_rows;
        }
    }
    row_ind.if_initialized = true;
    row_ind.if_zero = false;

    //////////

    std::vector<uint64_t> col_label(block_ncols, 0);
    for (uint64_t i = 0; i < nonzeros; ++i) {
        col_label[col_ptr[i]] = 1;
    }

    uint64_t non_empty_cols = 0;
    for (uint64_t i = 0; i < block_ncols; ++i) {
        if (col_label[i])
            ++non_empty_cols;
    }

    col_ind.alloc<I4>(non_empty_cols);
    auto col_ind_ptr = col_ind.get_aligned_ptr<I4>();

    non_empty_cols = 0;
    for (uint64_t i = 0; i < block_ncols; ++i) {
        if (col_label[i]) {
            col_ind_ptr[non_empty_cols] = i;
            col_label[i] = non_empty_cols;
            ++non_empty_cols;
        }
    }
    col_ind.if_initialized = true;
    col_ind.if_zero = false;

    //////////
    // row vector resize

    non_empty_rows = 0;
    for (uint64_t i = 0; i < nrows; ++i) {
        if (row_ptr[i + 1] != row_ptr[non_empty_rows]) {
            row_ptr[non_empty_rows + 1] = row_ptr[i + 1];
            ++non_empty_rows;
        }
    }
    nrows = non_empty_rows;
    row.resize<I1>(nrows + 1);

    //////////
    // col vector reindex

    for (uint64_t i = 0; i < nonzeros; ++i)
        col_ptr[i] = col_label[col_ptr[i]];
    ncols = non_empty_cols;

    //    std::cout << "Offd block size : " << block_nrows << " " << block_ncols <<
    //            " \t real size : " << nrows << " " << ncols << " \n";

    //////////

    if_indexed = true;
}

} // namespace matrix
} // namespace XAMG
