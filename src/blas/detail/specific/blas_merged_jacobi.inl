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
namespace blas {
namespace specific {

template <typename T, const uint16_t NV>
inline void merged_jacobi(vector::vector &x, const vector::vector &b,
                          const vector::vector &inv_diag, const vector::vector &r,
                          const vector::vector &relax_factor) {
    if (x.if_empty) {
        x.if_initialized = true;
        x.if_zero = false;
        return;
    }
    x.check(vector::vector::allocated);
    b.check(vector::vector::initialized);
    inv_diag.check(vector::vector::initialized);
    r.check(vector::vector::initialized);

    assert(!b.if_zero);
    assert(!inv_diag.if_zero);
    assert(inv_diag.nv == 1);

    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT b_ptr = b.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT inv_diag_ptr = inv_diag.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT relax_ptr = relax_factor.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    //////////

    if (x.if_zero && r.if_zero) {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = core_offset; i < core_offset + core_size; i++) {
            for (uint16_t nv = 0; nv < NV; ++nv) {
                x_ptr[i * NV + nv] = relax_ptr[nv] * inv_diag_ptr[i] * b_ptr[i * NV + nv];
            }
        }
    } else {
        if (x.if_zero)
            blas::forced_set_const<T, NV>(x, 0.0);
        if (r.if_zero)
            blas::forced_set_const<T, NV>(r, 0.0);

        XAMG_VECTOR_ALIGN
        for (uint64_t i = core_offset; i < core_offset + core_size; i++) {
            for (uint16_t nv = 0; nv < NV; ++nv) {
                x_ptr[i * NV + nv] +=
                    relax_ptr[nv] * inv_diag_ptr[i] * (b_ptr[i * NV + nv] - r_ptr[i * NV + nv]);
            }
        }
    }

    x.if_initialized = true;
    x.if_zero = false;

    perf.mem_read(4);
    perf.mem_write(1);
    perf.flop(6);
    XAMG_PERF_PRINT_DEBUG_INFO
}

} // namespace specific
} // namespace blas
} // namespace XAMG
