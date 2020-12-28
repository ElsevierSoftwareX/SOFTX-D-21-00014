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

#include "xamg_headers.h"
#include "xamg_types.h"

#include "primitives/vector/vector.h"

// dot_global:
#include "comm/mpi_wrapper.h"

// rand:
#include "misc/rand.h"

extern XAMG::perf_info perf;

/////////

namespace XAMG {

namespace blas {

// axpby: y = a*x + b*y
// copy:  y = x

template <typename T, const uint16_t NV>
inline void copy(const vector::vector &x, vector::vector &y);

template <typename T, const uint16_t NV>
inline void dot_global(const vector::vector &x, const vector::vector &y, vector::vector &res,
                       bool force = false);

template <typename T, const uint16_t NV>
inline void dot(const vector::vector &x, const vector::vector &y, vector::vector &res,
                bool force = false);

template <typename T, const uint16_t NV>
inline void vector_norm(const vector::vector &x, const uint16_t &norm_type, vector::vector &res,
                        bool force = false);

template <typename T, const uint16_t NV>
inline void set_const(vector::vector &x, const vector::vector &val, bool force = false);

template <typename T, const uint16_t NV>
inline void set_rand(vector::vector &x, bool true_rand = false);

template <typename T, const uint16_t NV>
inline void ax_y(const vector::vector &a, const vector::vector &x, vector::vector &y);

template <typename T, const uint16_t NV>
inline void axpby(const vector::vector &a, const vector::vector &x, const vector::vector &b,
                  vector::vector &y);

template <typename T, const uint16_t NV>
inline void scal(const vector::vector &x, vector::vector &y);

template <typename T, typename I, const uint16_t NV>
inline void gather(const vector::vector &x, const vector::indx_vector &indx, vector::vector &y,
                   bool force = true);

template <typename T, const uint16_t NV>
inline void forced_set_const(const vector::vector &x, const vector::vector &val);

//////////
//  extension:

template <typename T, const uint16_t NV>
inline void axpby_z(const vector::vector &a, const vector::vector &x, const vector::vector &b,
                    const vector::vector &y, vector::vector &z);

template <typename T, const uint16_t NV>
inline void axpbypcz(const vector::vector &a, const vector::vector &x, const vector::vector &b,
                     const vector::vector &y, const vector::vector &c, vector::vector &z);

template <typename T, const uint16_t NV>
inline void axpby(const T a, const vector::vector &x, const T b, vector::vector &y);

template <typename T, const uint16_t NV>
inline void xdivy_z(const vector::vector &x, const vector::vector &y, vector::vector &z);

template <typename T, const uint16_t NV>
inline void upload(vector::vector &x, const T *ptr, uint64_t proc_size, uint64_t proc_offset);

template <typename T, const uint16_t NV>
inline void pwr(vector::vector &x, const float64_t deg);

//////////
//  wrappers:

template <typename T>
inline void set_const(vector::vector &x, const T a);

template <typename T, const uint16_t NV>
inline void set_const(vector::vector &x, const T val, bool force = false);

template <typename T, const uint16_t NV>
inline void set_const(vector::vector &x, const std::vector<T> &a, bool force = false);

template <typename T, const uint16_t NV>
inline void forced_set_const(const vector::vector &x, const T val);

template <typename T, const uint16_t NV>
inline void axpby(const T a, const vector::vector &x, const T b, vector::vector &y);

struct vecs_t {
    vector::vector a0, a1, a_1;
    bool initialized = false;
};

template <typename F>
struct ConstVectorsCache {
    static std::map<int, vecs_t> cache;
    static vecs_t &alloc(int nv) {
        auto &c = cache[nv];
        if (!c.initialized) {
            vector::vector &a0 = c.a0;
            vector::vector &a1 = c.a1;
            vector::vector &a_1 = c.a_1;
            a0.alloc<F>(1, nv);
            a1.alloc<F>(1, nv);
            a_1.alloc<F>(1, nv);
            blas::set_const<F>(a0, 0.0);
            blas::set_const<F>(a1, 1.0);
            blas::set_const<F>(a_1, -1.0);
            c.initialized = true;
        }
        return c;
    }
    static const vector::vector &get_zeroes_vec(int nv) { return alloc(nv).a0; }
    static const vector::vector &get_ones_vec(int nv) { return alloc(nv).a1; }
    static const vector::vector &get_minus_ones_vec(int nv) { return alloc(nv).a_1; }
};

} // namespace blas

namespace vector {

template <typename F, uint16_t NV>
void construct_distributed(const std::shared_ptr<XAMG::part::part> part,
                           const XAMG::vector::vector &local_vec,
                           XAMG::vector::vector &distributed_vec) {
    distributed_vec.alloc<F>(part->numa_layer.block_size[id.nd_numa], NV);
    distributed_vec.set_part(part);
    XAMG::blas::copy<F, NV>(local_vec, distributed_vec);
}

} // namespace vector
} // namespace XAMG

#include "detail/blas.inl"
#include "detail/blas_wrapper.inl"
#include "detail/blas_ext.inl"
