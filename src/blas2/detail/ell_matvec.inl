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
namespace blas2 {
namespace backend {

template <typename F, typename I, uint16_t NV>
void ell_Axpy(const matrix::ell_matrix<F, I> &m, const vector::vector &x, vector::vector &y,
              uint16_t nv) {
#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "function , fp: <" << DEMANGLE_TYPEID_NAME(F) << ">; I: <"
              << DEMANGLE_TYPEID_NAME(I) << ">\n";
#endif
    assert(x.type_hash == y.type_hash);
    assert(m.val.type_hash == x.type_hash);
    m.col.check(vector::vector::initialized);
    m.val.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);

    auto col_ptr = m.col.template get_aligned_ptr<I>();
    auto val_ptr = m.val.template get_aligned_ptr<F>();

    auto x_ptr = x.get_aligned_ptr<F>();
    auto y_ptr = y.get_aligned_ptr<F>();

    /////////

    //    std::cout << "Dummy ELL SpMV\n";
    if (NV == 0) {
        //    for (i=0..rows)
        //      ...
        //      for (v = 0; v < nv; v++)
        //        ...
    } else {
        //    for (i=0..rows)
        //      ...
        //      for (v = 0; v < NV; v++)
        //        ...
    }

    assert(NV > 0);

    F val_j;
    uint64_t row_shift, col_shift;

    for (uint64_t i = 0; i < m.nrows; i++) {
        row_shift = i * NV;

        for (uint16_t j = 0; j < m.width; j++) {
            col_shift = col_ptr[i * m.width + j] * NV;
            val_j = val_ptr[i * m.width + j];

            // XAMG_VECTOR_NODEP
            // XAMG_VECTOR_ALIGN
            for (uint16_t m = 0; m < NV; ++m)
                y_ptr[row_shift + m] += x_ptr[col_shift + m] * val_j;
        }
    }

    y.if_zero = false;
}

template <typename F, typename I, uint16_t NV>
void ell_Ax_y(const matrix::ell_matrix<F, I> &m, const vector::vector &x, vector::vector &y,
              uint16_t nv) {
#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "function , fp: <" << DEMANGLE_TYPEID_NAME(F) << ">; I: <"
              << DEMANGLE_TYPEID_NAME(I) << ">\n";
#endif
    assert(x.type_hash == y.type_hash);
    assert(m.val.type_hash == x.type_hash);
    m.col.check(vector::vector::initialized);
    m.val.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    y.check(vector::vector::allocated);

    auto col_ptr = m.col.template get_aligned_ptr<I>();
    auto val_ptr = m.val.template get_aligned_ptr<F>();

    auto x_ptr = x.get_aligned_ptr<F>();
    auto y_ptr = y.get_aligned_ptr<F>();

    /////////

    //    std::cout << "Dummy ELL SpMV\n";

    if (NV == 0) {
        //    for (i=0..rows)
        //      ...
        //      for (v = 0; v < nv; v++)
        //        ...
    } else {
        //    for (i=0..rows)
        //      ...
        //      for (v = 0; v < NV; v++)
        //        ...
    }

    F val_j;
    uint64_t row_shift, col_shift;

    for (uint64_t i = 0; i < m.nrows; i++) {
        row_shift = i * NV;

        for (uint16_t k = 0; k < NV; ++k)
            y_ptr[row_shift + k] = 0.0;

        for (uint16_t j = 0; j < m.width; j++) {
            col_shift = col_ptr[i * m.width + j] * NV;
            val_j = val_ptr[i * m.width + j];

            // XAMG_VECTOR_NODEP
            // XAMG_VECTOR_ALIGN
            for (uint16_t m = 0; m < NV; ++m)
                y_ptr[row_shift + m] += x_ptr[col_shift + m] * val_j;
        }
    }

    y.if_initialized = true;
    y.if_zero = false;
}

} // namespace backend
} // namespace blas2
} // namespace XAMG
