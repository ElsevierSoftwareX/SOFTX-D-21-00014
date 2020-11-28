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
bool dense_matrix<F>::alloc() {
    //    assert((alloc_mode == mem::LOCAL) || (alloc_mode == mem::SHARED_NUMA));
    uint64_t nonzeros = nrows * ncols;
    if (nonzeros) {
        if_empty = false;

        val.alloc<F>(nonzeros);

        if (if_indexed) {
            row_ind.alloc<uint32_t>(nrows);
            col_ind.alloc<uint32_t>(ncols);
        }
    }

    return true;
}

template <typename F>
void dense_matrix<F>::compress() {
    //  dummy compression
    assert(0);

    //    XAMG::out << "Dummy compression for dense matrix..." << std::endl;
    //
    //    uint64_t non_empty_rows = nrows;
    //    uint64_t non_empty_cols = ncols;
    //
    //
    //    row_ind.alloc<uint32_t>(non_empty_rows);
    //    row_ind.check(vector::vector::allocated);
    //    auto row_ind_ptr = row_ind.get_aligned_ptr<uint32_t>();
    //
    //    for (uint64_t i = 0; i < nrows; ++i)
    //        row_ind_ptr[i] = i;
    //    row_ind.if_initialized = true;
    //    row_ind.if_zero = false;
    //
    ////////////
    //
    //    col_ind.alloc<uint32_t>(non_empty_cols, mem::NUMA);
    //    col_ind.check(vector::vector::allocated);
    //    auto col_ind_ptr = col_ind.get_aligned_ptr<uint32_t>();
    //
    //    for (uint64_t i = 0; i < ncols; ++i)
    //        col_ind_ptr[i] = i;
    //    col_ind.if_initialized = true;
    //    col_ind.if_zero = false;
    //
    ////////////
    //
    //    if_indexed = true;
}

} // namespace matrix
} // namespace XAMG
