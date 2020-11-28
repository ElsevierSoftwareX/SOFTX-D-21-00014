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

template <typename F0, typename I01, typename I02, typename I03, typename I04, typename F,
          typename I1, typename I2, typename I3, typename I4>
void convert(const csr_matrix<F0, I01, I02, I03, I04> &mat_in,
             csr_matrix<F, I1, I2, I3, I4> &mat_out) {
    mat_out.nrows = mat_in.nrows;
    mat_out.ncols = mat_in.ncols;
    mat_out.nonzeros = mat_in.nonzeros;

    mat_out.block_nrows = mat_in.block_nrows;
    mat_out.block_ncols = mat_in.block_ncols;

    mat_out.block_row_offset = mat_in.block_row_offset;
    mat_out.block_col_offset = mat_in.block_col_offset;

    mat_out.if_indexed = mat_in.if_indexed;

    if (mat_out.sharing_mode == mem::NUMA) {
        mpi::bcast<uint64_t>(&mat_out.nrows, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.ncols, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.nonzeros, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.block_nrows, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.block_ncols, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.block_row_offset, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.block_col_offset, 1, 0, mpi::INTRA_NUMA);
        uint8_t _ind = mat_out.if_indexed;
        mpi::bcast<uint8_t>(&_ind, 1, 0, mpi::INTRA_NUMA);
        mat_out.if_indexed = _ind;
    }

    mat_out.alloc();

    /////////

    if (!mat_out.if_empty) {
        vector::convert<I01, I1>(mat_in.row, mat_out.row);
        vector::convert<I02, I2>(mat_in.col, mat_out.col);
        vector::convert<F0, F>(mat_in.val, mat_out.val);

        if (mat_out.sharing_mode == mem::NUMA) {
            bcast_vector_state(mat_out.row, mpi::INTRA_NUMA);
            bcast_vector_state(mat_out.col, mpi::INTRA_NUMA);
            bcast_vector_state(mat_out.val, mpi::INTRA_NUMA);
        }

        if (mat_out.if_indexed) {
            vector::convert<I03, I3>(mat_in.row_ind, mat_out.row_ind);
            vector::convert<I04, I4>(mat_in.col_ind, mat_out.col_ind);

            if (mat_out.sharing_mode == mem::NUMA) {
                bcast_vector_state(mat_out.row_ind, mpi::INTRA_NUMA);
                bcast_vector_state(mat_out.col_ind, mpi::INTRA_NUMA);
            }
        }
    }
}

template <typename F0, typename F>
void convert(const dense_matrix<F0> &mat_in, dense_matrix<F> &mat_out) {
    mat_out.nrows = mat_in.nrows;
    mat_out.ncols = mat_in.ncols;

    mat_out.block_nrows = mat_in.block_nrows;
    mat_out.block_ncols = mat_in.block_ncols;

    mat_out.block_row_offset = mat_in.block_row_offset;
    mat_out.block_col_offset = mat_in.block_col_offset;

    mat_out.if_indexed = mat_in.if_indexed;

    mat_out.alloc();

    /////////

    if (!mat_out.if_empty) {
        vector::convert<F0, F>(mat_in.val, mat_out.val);

        if (mat_out.if_indexed) {
            vector::convert<uint32_t, uint32_t>(mat_in.row_ind, mat_out.row_ind);
            vector::convert<uint32_t, uint32_t>(mat_in.col_ind, mat_out.col_ind);
        }
    }
}

} // namespace matrix
} // namespace XAMG
