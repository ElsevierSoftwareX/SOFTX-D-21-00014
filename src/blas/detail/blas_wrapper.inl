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

template <typename T, const uint16_t NV>
inline void blas::set_const(vector::vector &x, const T a, bool force) {

    vector::vector vec;
    auto vec_ptr = vec.alloc_and_get_aligned_ptr<T>(1, NV);

    if (fabs(a) > 1.e-100)
        vec.if_zero = false;
    for (size_t m = 0; m < NV; ++m)
        vec_ptr[m] = a;
    vec.if_initialized = true;

    blas::set_const<T, NV>(x, vec, force);
}

template <typename T, const uint16_t NV>
inline void blas::set_const(vector::vector &x, const std::vector<T> &a, bool force) {

    vector::vector vec;
    auto vec_ptr = vec.alloc_and_get_aligned_ptr<T>(1, NV);

    assert(a.size() == NV);
    for (size_t m = 0; m < NV; ++m)
        vec_ptr[m] = a[m];
    vec.if_zero = false;
    vec.if_initialized = true;

    blas::set_const<T, NV>(x, vec, force);
}

//////////

template <typename T, const uint16_t NV>
inline void blas::forced_set_const(const vector::vector &x, const T a) {

    vector::vector vec;
    auto vec_ptr = vec.alloc_and_get_aligned_ptr<T>(1, NV);

    if (fabs(a) > 1.e-100)
        vec.if_zero = false;
    for (size_t m = 0; m < NV; ++m)
        vec_ptr[m] = a;
    vec.if_initialized = true;

    blas::forced_set_const<T, NV>(x, vec);
}

//////////

template <typename T, const uint16_t NV>
inline void blas::axpby(const T a, const vector::vector &x, const T b, vector::vector &y) {

    vector::vector vec1, vec2;
    auto vec1_ptr = vec1.alloc_and_get_aligned_ptr<T>(1, NV);
    auto vec2_ptr = vec2.alloc_and_get_aligned_ptr<T>(1, NV);

    if (fabs(a) > 1.e-100)
        vec1.if_zero = false;
    for (size_t m = 0; m < NV; ++m)
        vec1_ptr[m] = a;
    vec1.if_initialized = true;

    if (fabs(b) > 1.e-100)
        vec2.if_zero = false;
    for (size_t m = 0; m < NV; ++m)
        vec2_ptr[m] = b;
    vec2.if_initialized = true;

    blas::axpby<T, NV>(vec1, x, vec2, y);
}

} // namespace XAMG
