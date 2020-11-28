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

#include "misc/reorder.h"

namespace XAMG {
namespace matrix {

template <typename F, typename I1, typename I2, typename I3, typename I4>
void distribute(const csr_matrix<F, I1, I2, I3, I4> &mat_in, csr_matrix<F, I1, I2, I3, I4> &mat_out,
                const part::part_layer &layer, mpi::scope comm);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void distribute(std::shared_ptr<csr_matrix<F, I1, I2, I3, I4>> &mat, const part::part_layer &layer,
                mpi::scope comm);

template <typename F>
void distribute(const dense_matrix<F> &mat_in, dense_matrix<F> &mat_out,
                const part::part_layer &layer, mpi::scope comm);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void collect(const csr_matrix<F, I1, I2, I3, I4> &mat_in, csr_matrix<F, I1, I2, I3, I4> &mat_out,
             mpi::scope comm);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void collect(std::shared_ptr<csr_matrix<F, I1, I2, I3, I4>> &mat, mpi::scope comm);

template <typename F>
void collect(const dense_matrix<F> &mat_in, dense_matrix<F> &mat_out, mpi::scope comm);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void redistribute(std::shared_ptr<csr_matrix<F, I1, I2, I3, I4>> &mat, const map_info &mapping_info,
                  const segment::hierarchy &layer);

////

template <typename F, typename I1, typename I2, typename I3, typename I4>
void split_by_rows(const csr_matrix<F, I1, I2, I3, I4> &mat,
                   std::vector<csr_matrix<F, I1, I2, I3, I4>> &blocks,
                   const part::part_layer &layer);

template <typename F>
void split_by_rows(const dense_matrix<F> &mat, std::vector<dense_matrix<F>> &blocks,
                   const part::part_layer &layer);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void split_by_columns(const csr_matrix<F, I1, I2, I3, I4> &mat,
                      std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &mat_chunks,
                      const part::part_layer &layer);

template <typename F>
void merge_row_blocks(const std::vector<dense_matrix<F>> &blocks,
                      dense_matrix<F> &mat); // merge row of blocks

template <typename F>
void merge_col_blocks(const std::vector<dense_matrix<F>> &blocks,
                      dense_matrix<F> &mat); // merge column of blocks

////

template <typename F, typename I1, typename I2, typename I3, typename I4>
void transpose(const csr_matrix<F, I1, I2, I3, I4> &mat, csr_matrix<F, I1, I2, I3, I4> &matT);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void transpose(const std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks,
               std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csrT_chunks);

////

template <typename F>
void inverse(const dense_matrix<F> &mat, dense_matrix<F> &mat_inv);

template <typename F>
void inverse(const matrix &A, matrix &A_inv);

////

template <typename F0, typename I01, typename I02, typename I03, typename I04, typename F,
          typename I1, typename I2, typename I3, typename I4>
void convert(const csr_matrix<F0, I01, I02, I03, I04> &mat_in,
             csr_matrix<F, I1, I2, I3, I4> &mat_out);

template <typename F0, typename F>
void convert(const dense_matrix<F0> &mat_in, dense_matrix<F> &mat_out);

} // namespace matrix
} // namespace XAMG

#include "operations/transform.inl"
#include "operations/segment.inl"
#include "operations/transpose.inl"

#include "operations/convert.inl"

#include "operations/inverse.inl"
