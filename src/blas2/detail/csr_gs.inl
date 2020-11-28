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

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
void csr_SGS(const matrix::csr_matrix<F, I1, I2, I3, I4> &m, const vector::vector &inv_diag,
             const vector::vector &t, vector::vector &x, const vector::vector &relax_factor,
             uint16_t nv) {
#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "function , fp: <" << DEMANGLE_TYPEID_NAME(F) << ">; I1: <"
              << DEMANGLE_TYPEID_NAME(I1) << ">; I2: <" << DEMANGLE_TYPEID_NAME(I2) << ">; I3: <"
              << DEMANGLE_TYPEID_NAME(I3) << ">; I4: <" << DEMANGLE_TYPEID_NAME(I4) << ">\n";
#endif
    assert(x.type_hash == t.type_hash);
    assert(m.val.type_hash == x.type_hash);
    m.row.check(vector::vector::initialized);
    m.col.check(vector::vector::initialized);
    m.val.check(vector::vector::initialized);
    inv_diag.check(vector::vector::initialized);
    t.check(vector::vector::initialized);
    x.check(vector::vector::initialized);

    const I1 *XAMG_RESTRICT row_ptr = m.row.template get_aligned_ptr<I1>();
    const I2 *XAMG_RESTRICT col_ptr = m.col.template get_aligned_ptr<I2>();
    const F *XAMG_RESTRICT val_ptr = m.val.template get_aligned_ptr<F>();

    const F *XAMG_RESTRICT inv_diag_ptr = inv_diag.get_aligned_ptr<F>();
    const F *XAMG_RESTRICT t_ptr = t.get_aligned_ptr<F>();
    F *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<F>();

    const F *XAMG_RESTRICT relax_ptr = relax_factor.get_aligned_ptr<F>();

    /////////

    assert(m.sharing_mode == mem::CORE);
    assert(!m.if_indexed);

    assert((x.sharing_mode == mem::NUMA_NODE) && (t.sharing_mode == mem::NUMA_NODE));

    //    if (NV == 0) {
    //    } else
    {
        uint64_t core_offset = m.block_row_offset - x.global_numa_offset();
        uint64_t core_size = m.nrows;
        XAMG_STACK_ALIGNMENT_PREFIX F sum[NV];
        XAMG_STACK_ALIGNMENT_PREFIX F coeff[NV];

        //        Forward
        uint64_t row_shift, col_shift;
        for (uint64_t i = 0; i < core_size; i++) {
            row_shift = (i + core_offset) * NV;
            XAMG_VECTOR_ALIGN
            for (uint16_t m = 0; m < NV; m++)
                sum[m] = t_ptr[row_shift + m];

            XAMG_NOVECTOR
            for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                col_shift = (col_ptr[j] + core_offset) * NV;

                XAMG_VECTOR_ALIGN
                for (uint16_t m = 0; m < NV; m++)
                    sum[m] -= x_ptr[col_shift + m] * val_ptr[j];
            }

            XAMG_VECTOR_NODEP
            XAMG_VECTOR_ALIGN
            for (uint16_t m = 0; m < NV; m++)
                coeff[m] = relax_ptr[m] * inv_diag_ptr[i + core_offset];
            XAMG_VECTOR_NODEP
            XAMG_VECTOR_ALIGN
            for (uint16_t m = 0; m < NV; m++)
                x_ptr[row_shift + m] += coeff[m] * sum[m];
        }

        /////////
        //        Backward

        for (uint64_t ii = 0; ii < core_size; ii++) {
            uint64_t i = core_size - 1 - ii;
            row_shift = (i + core_offset) * NV;

            XAMG_VECTOR_ALIGN
            for (uint16_t m = 0; m < NV; m++)
                sum[m] = t_ptr[row_shift + m];

            XAMG_NOVECTOR
            for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                col_shift = (col_ptr[j] + core_offset) * NV;

                XAMG_VECTOR_ALIGN
                for (uint16_t m = 0; m < NV; m++)
                    sum[m] -= x_ptr[col_shift + m] * val_ptr[j];
            }

            XAMG_VECTOR_NODEP
            XAMG_VECTOR_ALIGN
            for (uint16_t m = 0; m < NV; m++)
                coeff[m] = relax_ptr[m] * inv_diag_ptr[i + core_offset];
            XAMG_VECTOR_NODEP
            XAMG_VECTOR_ALIGN
            for (uint16_t m = 0; m < NV; m++)
                x_ptr[row_shift + m] += coeff[m] * sum[m];
        }
    }

    x.if_zero = false;
}

} // namespace backend
} // namespace blas2
} // namespace XAMG
