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

template <typename F, uint16_t NV>
void dense_Axpy(const matrix::dense_matrix<F> &m, const vector::vector &x, vector::vector &y,
                uint16_t nv) {
#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "function , fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
    if (m.if_empty) {
        y.if_zero = false;
        return;
    }
    assert(x.type_hash == y.type_hash);
    assert(m.val.type_hash == x.type_hash);
    m.val.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);

    assert((x.nv == y.nv) && (x.nv == NV));
    assert(m.ncols == x.size);

    const F *XAMG_RESTRICT val_ptr = m.val.template get_aligned_ptr<F>();

    const F *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<F>();
    F *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<F>();

    /////////
    XAMG_STACK_ALIGNMENT_PREFIX F sum[NV];

    for (uint64_t i = 0; i < m.nrows; ++i) {
        XAMG_VECTOR_ALIGN
        for (uint64_t j = 0; j < m.ncols; ++j)
            for (uint16_t k = 0; k < NV; ++k)
                y_ptr[i * NV + k] += val_ptr[i * m.ncols + j] * x_ptr[j * NV + k];
    }

    y.if_zero = false;
}

////////////////////

template <typename F, uint16_t NV>
void dense_Ax_y(const matrix::dense_matrix<F> &m, const vector::vector &x, vector::vector &y,
                uint16_t nv) {
#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "function , fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
    //    performed by only numa (or even node?) master process!!!
    if (m.if_empty) {
        y.if_initialized = true;
        y.if_zero = false;
        return;
    }
    assert(x.type_hash == y.type_hash);
    assert(m.val.type_hash == x.type_hash);
    m.val.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    y.check(vector::vector::allocated);

    if (x.if_zero) {
        blas::set_const<F, NV>(y, 0.0, true);
        return;
    }

    assert((x.nv == y.nv) && (x.nv == NV));
    assert(m.ncols == x.size);

    const F *XAMG_RESTRICT val_ptr = m.val.template get_aligned_ptr<F>();

    const F *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<F>();
    F *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<F>();

    //    uint64_t core_size, core_offset;
    //    y.get_core_range<T>(core_size, core_offset);
    //    XAMG::out << ALLRANKS << m.nrows << " :: " << core_offset << " : " << core_size <<
    //    std::endl; XAMG::out << ALLRANKS << m.nrows << std::endl;

    /////////

    for (uint64_t i = 0; i < m.nrows; ++i) {
        for (uint16_t k = 0; k < NV; ++k)
            y_ptr[i * NV + k] = 0.0;

        XAMG_VECTOR_ALIGN
        for (uint64_t j = 0; j < m.ncols; ++j)
            for (uint16_t k = 0; k < NV; ++k)
                y_ptr[i * NV + k] += val_ptr[i * m.ncols + j] * x_ptr[j * NV + k];
    }

    y.if_initialized = true;
    y.if_zero = false;
}

} // namespace backend
} // namespace blas2
} // namespace XAMG
