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
inline void blas::axpby_z(const vector::vector &a, const vector::vector &x, const vector::vector &b,
                          const vector::vector &y, vector::vector &z) {
    if (a.if_zero || x.if_zero) {
        blas::ax_y<T, NV>(b, y, z);
        return;
    }

    if (b.if_zero || y.if_zero) {
        blas::ax_y<T, NV>(a, x, z);
        return;
    }

    x.check(vector::vector::allocated);
    y.check(vector::vector::allocated);
    z.check(vector::vector::allocated);

    if (x.if_empty && y.if_empty && z.if_empty) {
        z.if_initialized = true;
        z.if_zero = false;
        return;
    }

    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);
    a.check(vector::vector::initialized);
    b.check(vector::vector::initialized);

    assert(x.size == y.size);
    assert(x.size == z.size);
    assert(a.size == b.size);
    assert(a.nv == b.nv);

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();
    T *XAMG_RESTRICT z_ptr = z.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT a_ptr = a.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT b_ptr = b.get_aligned_ptr<T>();

    assert(x_ptr != y_ptr);
    assert(x_ptr != z_ptr);

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    /////////
    assert(a.size == 1);

    if (a.nv == 1) {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                z_ptr[i + nv] = a_ptr[0] * x_ptr[i + nv] + b_ptr[0] * y_ptr[i + nv];
        }
    } else {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                z_ptr[i + nv] = a_ptr[nv] * x_ptr[i + nv] + b_ptr[nv] * y_ptr[i + nv];
        }
    }

    z.if_initialized = true;
    z.if_zero = false;

    if (z.size > 1) {
        perf.mem_read(2);
        perf.mem_write(1);
        perf.flop(3);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

//////////

template <typename T, const uint16_t NV>
inline void blas::axpbypcz(const vector::vector &a, const vector::vector &x,
                           const vector::vector &b, const vector::vector &y,
                           const vector::vector &c, vector::vector &z) {
    if (a.if_zero || x.if_zero) {
        blas::axpby<T, NV>(b, y, c, z);
        return;
    }

    if (b.if_zero || y.if_zero) {
        blas::axpby<T, NV>(a, x, c, z);
        return;
    }

    if (c.if_zero || z.if_zero) {
        blas::axpby_z<T, NV>(a, x, b, y, z);
        return;
    }

    x.check(vector::vector::allocated);
    y.check(vector::vector::allocated);
    z.check(vector::vector::allocated);

    if (x.if_empty && y.if_empty && z.if_empty) {
        z.if_zero = false;
        return;
    }

    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);
    z.check(vector::vector::initialized);
    a.check(vector::vector::initialized);
    b.check(vector::vector::initialized);
    c.check(vector::vector::initialized);

    assert(x.size == y.size);
    assert(x.size == z.size);
    assert(a.size == b.size);
    assert(a.size == c.size);
    assert(a.nv == b.nv);
    assert(a.nv == c.nv);

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();
    T *XAMG_RESTRICT z_ptr = z.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT a_ptr = a.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT b_ptr = b.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT c_ptr = c.get_aligned_ptr<T>();

    assert(x_ptr != y_ptr);
    assert(x_ptr != z_ptr);

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    /////////
    assert(a.size == 1);

    if (a.nv == 1) {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                z_ptr[i + nv] =
                    a_ptr[0] * x_ptr[i + nv] + b_ptr[0] * y_ptr[i + nv] + c_ptr[0] * z_ptr[i + nv];
        }
    } else {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                z_ptr[i + nv] = a_ptr[nv] * x_ptr[i + nv] + b_ptr[nv] * y_ptr[i + nv] +
                                c_ptr[nv] * z_ptr[i + nv];
        }
    }

    z.if_zero = false;

    if (z.size > 1) {
        perf.mem_read(3);
        perf.mem_write(1);
        perf.flop(5);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

/////////

template <typename T, const uint16_t NV>
inline void blas::xdivy_z(const vector::vector &x, const vector::vector &y, vector::vector &z) {
    if (x.if_zero) {
        blas::set_const<T, NV>(z, 0.0);
        return;
    }

    if (y.if_zero) {
        assert("Div by 0!\n" && 0);
        return;
    }

    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);
    z.check(vector::vector::allocated);
    assert(x.nv == y.nv);
    assert(x.nv == z.nv);

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();
    T *XAMG_RESTRICT z_ptr = z.get_aligned_ptr<T>();

    assert(x_ptr != y_ptr);
    assert(x_ptr != z_ptr);

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    /////////
    assert(x.size == y.size);
    assert(x.size == z.size);

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); ++i) {
        z_ptr[i] = x_ptr[i] / y_ptr[i];
    }

    z.if_initialized = true;
    z.if_zero = false;

    if (z.size > 1) {
        perf.mem_read(2);
        perf.mem_write(1);
        perf.flop(1);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

/////////

template <typename T, const uint16_t NV>
inline void blas::upload(vector::vector &x, const T *ptr, uint64_t proc_size,
                         uint64_t proc_offset) {
    x.check(vector::vector::allocated);
    assert(x.nv == NV);
    assert(x.sharing_mode == mem::NUMA_NODE);
    if (x.if_empty) {
        x.if_initialized = true;
        x.if_zero = false;
        return;
    }

    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();

    uint64_t core_size = proc_size;
    uint64_t core_offset = proc_offset - x.global_numa_offset();

    /////////
    /*
    //XAMG_VECTOR_ALIGN
        for(uint64_t i = NV*core_offset; i < NV*(core_offset + core_size); i+=NV) {
            uint64_t ii = i - NV*core_offset;
            for (uint16_t nv = 0; nv < NV; ++nv)
                x_ptr[i+nv] = ptr[ii+nv];
        }
    */

    uint64_t offset = NV * core_offset;
    for (uint64_t i = 0; i < NV * core_size; ++i) {
        x_ptr[i + offset] = ptr[i];
    }

    mpi::barrier(mpi::INTRA_NUMA);

    x.if_initialized = true;
    x.if_zero = false;

    if (x.size > 1) {
        perf.mem_read(1);
        perf.mem_write(1);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

/////////

template <typename T, const uint16_t NV>
inline void blas::pwr(vector::vector &x, const float64_t deg) {
    if (x.if_empty)
        return;
    x.check(vector::vector::initialized);

    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    /////////

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); ++i) {
        x_ptr[i] = pow(x_ptr[i], deg);
    }

    if (x.size > 1) {
        perf.mem_read(1);
        perf.mem_write(1);
        perf.flop(1);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

} // namespace XAMG
