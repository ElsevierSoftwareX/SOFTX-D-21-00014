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
inline void blas::ax_y(const vector::vector &a, const vector::vector &x, vector::vector &y) {
    if (a.if_zero || x.if_zero) {
        y.if_initialized = true;
        y.if_zero = true;
        return;
    }

    if (x.if_empty && y.if_empty) {
        y.if_initialized = true;
        y.if_zero = false;
        return;
    }

    a.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    y.check(vector::vector::nonempty);

    assert(x.size == y.size);

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT a_ptr = a.get_aligned_ptr<T>();

    assert(x_ptr != y_ptr);

    uint64_t core_size, core_offset;
    y.get_core_range<T>(core_size, core_offset);

    /////////
    assert(a.size == 1);

    if (a.nv == 1) {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); ++i) {
            y_ptr[i] = a_ptr[0] * x_ptr[i];
        }
    } else {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                y_ptr[i + nv] = a_ptr[nv] * x_ptr[i + nv];
        }
    }

    y.if_initialized = true;
    y.if_zero = false;

    if (y.size > 1) {
        perf.mem_read(1);
        perf.mem_write(1);
        perf.flop(1);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

//////////

template <typename T, const uint16_t NV>
inline void blas::axpby(const vector::vector &a, const vector::vector &x, const vector::vector &b,
                        vector::vector &y) {
    if (a.if_zero || x.if_zero) {
        blas::scal<T, NV>(b, y);
        return;
    }
    if (b.if_zero || y.if_zero) {
        blas::ax_y<T, NV>(a, x, y);
        return;
    }

    x.check(vector::vector::allocated);
    y.check(vector::vector::allocated);
    if (x.if_empty && y.if_empty) {
        y.if_zero = false;
        return;
    }

    a.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);

    assert(x.size == y.size);
    assert(a.size == b.size);
    assert(a.nv == b.nv);

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT a_ptr = a.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT b_ptr = b.get_aligned_ptr<T>();

    assert(x_ptr != y_ptr);

    uint64_t core_size, core_offset;
    y.get_core_range<T>(core_size, core_offset);

    /////////
    assert(a.size == 1);

    if (a.nv == 1) {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); ++i) {
            y_ptr[i] = a_ptr[0] * x_ptr[i] + b_ptr[0] * y_ptr[i];
        }
    } else {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                y_ptr[i + nv] = a_ptr[nv] * x_ptr[i + nv] + b_ptr[nv] * y_ptr[i + nv];
        }
    }

    y.if_zero = false;

    if (y.size > 1) {
        perf.mem_read(2);
        perf.mem_write(1);
        perf.flop(3);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

//////////

template <typename T, const uint16_t NV>
inline void blas::scal(const vector::vector &x, vector::vector &y) {
    if (x.if_zero || y.if_zero) {
        y.if_zero = true;
        return;
    }

    x.check(vector::vector::allocated);
    y.check(vector::vector::allocated);
    if (x.if_empty && y.if_empty)
        return;
    if (y.if_empty && (x.size == 1))
        return;

    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();

    assert(x_ptr != y_ptr);

    uint64_t core_size, core_offset;
    y.get_core_range<T>(core_size, core_offset);

    /////////

    if (x.nv == 1) {
        if (x.size == 1) {
            XAMG_VECTOR_ALIGN
            for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); ++i)
                y_ptr[i] *= x_ptr[0];
        } else {
            assert(x.size == y.size);
            XAMG_VECTOR_ALIGN
            for (uint64_t i = core_offset; i < core_offset + core_size; ++i) {
                for (uint16_t nv = 0; nv < NV; ++nv)
                    y_ptr[i * NV + nv] *= x_ptr[i];
            }
        }
    } else {
        assert(y.nv == x.nv);
        if (x.size == 1) {
            XAMG_VECTOR_ALIGN
            for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
                for (uint16_t nv = 0; nv < NV; ++nv)
                    y_ptr[i + nv] *= x_ptr[nv];
            }
        } else {
            assert(x.size == y.size);
            XAMG_VECTOR_ALIGN
            for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); ++i) {
                y_ptr[i] *= x_ptr[i];
            }
        }
    }

    if (y.size > 1) {
        if (x.size > 1)
            perf.mem_read(1);
        perf.mem_write(1);
        perf.flop(1);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

//////////

template <typename T, const uint16_t NV>
inline void blas::copy(const vector::vector &x, vector::vector &y) {
    if (x.if_zero) {
        y.if_initialized = true;
        y.if_zero = x.if_zero;
        return;
    }

    x.check(vector::vector::allocated);
    y.check(vector::vector::allocated);
    //    if (x.if_empty && y.if_empty) return;
    if (x.if_empty) {
        y.if_initialized = true;
        y.if_zero = x.if_zero; // false
        return;
    }

    x.check(vector::vector::initialized);
    y.check(vector::vector::nonempty);

    assert(x.nv == y.nv);

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();

    assert(x_ptr != y_ptr);

    ////
    if (x.sharing_mode == y.sharing_mode) {
        assert(x.size == y.size);
        uint64_t core_size, core_offset;
        x.get_core_range<T>(core_size, core_offset);

        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); ++i)
            y_ptr[i] = x_ptr[i];
    } else {
        uint64_t block_size = 0;
        uint64_t block_offset = 0;
        if ((x.sharing_mode == mem::CORE) && (y.sharing_mode == mem::NUMA_NODE)) {
            // local vector core offset can differ from shared/distributed vector core offset!
            block_size = x.size;
            block_offset = x.ext_offset - y.global_numa_offset();
            // XAMG::out << ALLRANKS << block_offset << " : " << y.intra_numa_offset() << std::endl;
        } else if ((x.sharing_mode == mem::NUMA_NODE) && (y.sharing_mode == mem::NODE)) {
            if (id.numa_master_process()) {
                block_size = x.size;
                block_offset = x.numa_offset();
            } else {
                block_size = 0;
                block_offset = 0;
            }
        } else {
            assert(0 && "Incorrect vector upload configuration");
        }
        /*
        //XAMG_VECTOR_ALIGN
                for(uint64_t i = NV*block_offset; i < NV*(block_offset + block_size); i+=NV) {
                    uint64_t ii = i - NV*block_offset;
        //XAMG_VECTOR_ALIGN
                    for (uint16_t nv = 0; nv < NV; ++nv)
                        y_ptr[i+nv] = x_ptr[ii+nv];
                }
        */
        uint64_t offset = NV * block_offset;
        XAMG_VECTOR_ALIGN
        for (uint64_t i = 0; i < NV * block_size; ++i) {
            y_ptr[i + offset] = x_ptr[i];
        }

        mpi::barrier(mpi::INTRA_NUMA); // alignment of local vectors can be differ from the one
                                       // of core_size blocks; of importance for the second IF case
    }

    /////////

    y.if_initialized = true;
    y.if_zero = false;

    if (y.size > 1) {
        perf.mem_read(1);
        perf.mem_write(1);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

//////////

template <typename T>
inline void blas::set_const(vector::vector &x, const T val) {
    x.check(vector::vector::allocated);
    if (x.if_empty) {
        x.if_initialized = true;
        x.if_zero = false;
        return;
    }

    x.check(vector::vector::nonempty);
    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    XAMG_VECTOR_ALIGN
    for (uint64_t i = x.nv * core_offset; i < x.nv * (core_offset + core_size); ++i) {
        x_ptr[i] = val;
    }

    x.if_initialized = true;
    x.if_zero = (fabs(val) < 1.e-100);
}

template <typename T, const uint16_t NV>
inline void blas::set_const(vector::vector &x, const vector::vector &val, bool force) {
    if (val.if_zero && (!force)) {
        x.if_initialized = true;
        x.if_zero = true;
        return;
    }

    x.check(vector::vector::allocated);
    if (x.if_empty) {
        x.if_initialized = true;
        x.if_zero = false;
        return;
    }

    x.check(vector::vector::nonempty);
    val.check(vector::vector::initialized);

    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT val_ptr = val.get_aligned_ptr<T>();

    assert(x_ptr != val_ptr);

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    //////////

    if ((val.size == 1) && (val.nv == 1)) {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); ++i) {
            x_ptr[i] = val_ptr[0];
        }
    } else if (val.nv == x.nv) {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                x_ptr[i + nv] = val_ptr[nv];
        }
    } else {
        assert(0);
    }

    x.if_initialized = true;
    x.if_zero = false;

    if (x.size > 1) {
        perf.mem_write(1);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

//////////

template <typename T, const uint16_t NV>
inline void blas::set_rand(vector::vector &x, bool rand) {
    x.check(vector::vector::allocated);
    if (x.if_empty) {
        x.if_initialized = true;
        x.if_zero = false;
        return;
    }

    x.check(vector::vector::nonempty);

    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    //////////
    //  <TODO>: add vectorizable version of this function

    if (rand == true) {
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                x_ptr[i + nv] = true_rand<T>() - 0.5;
        }
    } else {
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                x_ptr[i + nv] = pseudo_rand<T>(i + nv + NV * x.global_numa_offset()) - 0.5;
        }
    }

    x.if_initialized = true;
    x.if_zero = false;

    if (x.size > 1) {
        perf.mem_write(1);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

//////////

template <typename T, const uint16_t NV>
inline void blas::dot_global(const vector::vector &x, const vector::vector &y, vector::vector &res,
                             bool force) {
    x.check(vector::vector::allocated);
    y.check(vector::vector::allocated);

    res.check(vector::vector::nonempty);

    T *XAMG_RESTRICT res_ptr = res.get_aligned_ptr<T>();

    /////////

    //    blas::dot<T, NV>(x, y, res, force);
    blas::dot<T, NV>(x, y, res, true);

    //    if (!res.if_zero)
    {
        XAMG_STACK_ALIGNMENT_PREFIX T dot_global[NV];

        mpi::allreduce_sum<T>(res_ptr, dot_global, NV);

        XAMG_VECTOR_ALIGN
        for (uint16_t m = 0; m < NV; ++m)
            res_ptr[m] = dot_global[m];
        res.if_zero = false;

        perf.allreduce(1);
    }
    res.if_initialized = true;
}

//////////
// TODO: revise if we need "bool force" argument
template <typename T, const uint16_t NV>
inline void blas::dot(const vector::vector &x, const vector::vector &y, vector::vector &res,
                      bool force) {
    if (x.if_zero || y.if_zero) {
        res.if_initialized = true;
        res.if_zero = true;
        blas::set_const<T, NV>(res, 0, true);
        return;
    }

    x.check(vector::vector::allocated);
    y.check(vector::vector::allocated);
    res.check(vector::vector::allocated);
    if (x.if_empty && y.if_empty) {
        res.if_initialized = true;
        blas::set_const<T, NV>(res, 0, true);
        res.if_zero = false;
        return;
    }

    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();
    T *XAMG_RESTRICT res_ptr = res.get_aligned_ptr<T>();

    XAMG_STACK_ALIGNMENT_PREFIX T dot_global[NV];

    assert(x.nv == NV);
    assert(y.nv == NV);
    assert(res.nv == NV);
    assert(x.size == y.size);

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);
    //    XAMG::out << XAMG::ALLRANKS << "DOT  : " << core_offset << " " << core_size << std::endl;

    //////////

    for (uint16_t nv = 0; nv < NV; ++nv)
        res_ptr[nv] = 0.0;

    if (x_ptr == y_ptr) {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                res_ptr[nv] += x_ptr[i + nv] * x_ptr[i + nv];
        }
    } else {
        XAMG_VECTOR_ALIGN
        for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
            for (uint16_t nv = 0; nv < NV; ++nv)
                res_ptr[nv] += x_ptr[i + nv] * y_ptr[i + nv];
        }
    }

    res.if_initialized = true;
    res.if_zero = false;

    if (y.size > 1) {
        if (x_ptr == y_ptr)
            perf.mem_read(1);
        else
            perf.mem_read(2);
        perf.flop(2);
        XAMG_PERF_PRINT_DEBUG_INFO
    }
}

//////////
enum vector_norm_types { L2_norm, Linf_norm };

template <typename T, const uint16_t NV>
inline void blas::vector_norm(const vector::vector &x, const uint16_t &norm_type,
                              vector::vector &res, bool force) {
    switch (norm_type) {
    case L2_norm: {
        blas::dot_global<T, NV>(x, x, res, force);
        break;
    }
    case Linf_norm: {
        assert(0);
        //        blas::max<T, NV>(x, res, force);

        break;
    }
    default: {
        assert(0 && "Incorrect norm type specified");
    }
    }
}

//////////

template <typename T, typename I, const uint16_t NV>
inline void blas::gather(const vector::vector &x, const vector::indx_vector &indx,
                         vector::vector &y, bool force) {
    assert(y.size == indx.size);
    if (x.if_zero) {
        y.if_initialized = true;
        y.if_zero = true;
    }

    x.check(vector::vector::allocated);
    y.check(vector::vector::allocated);
    indx.check(vector::vector::allocated);

    if (x.if_empty || (y.if_empty && indx.if_empty)) {
        y.if_initialized = true;
        y.if_zero = x.if_zero;
        return;
    }

    x.check(vector::vector::initialized);
    y.check(vector::vector::nonempty);
    indx.check(vector::vector::initialized);

    assert(indx.type_hash == typeid(I).hash_code());

    const T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();
    const I *XAMG_RESTRICT i_ptr = indx.get_aligned_ptr<I>();
    assert(x_ptr != y_ptr);
    assert(x.sharing_mode == mem::NUMA_NODE);
    assert(y.sharing_mode == indx.sharing_mode);

    if (indx.sharing_mode == mem::CORE) {
        uint32_t offset = x.intra_numa_offset();

        if (x.if_zero) {
            XAMG_VECTOR_ALIGN
            for (uint64_t i = 0; i < indx.size * NV; ++i)
                y_ptr[i] = 0.0;
        } else {
            for (uint64_t i = 0; i < indx.size; ++i) {
                uint64_t i1 = i * NV;
                uint64_t i2 = (i_ptr[i] + offset) * NV;

                XAMG_VECTOR_ALIGN
                for (uint16_t m = 0; m < NV; m++)
                    y_ptr[i1 + m] = x_ptr[i2 + m];
            }
        }
    } else {
        uint32_t offset = 0;
        if (indx.sharing_mode == mem::NUMA) {
            offset = 0;
        } else if (indx.sharing_mode == mem::NODE) {
            offset = x.numa_offset();
        } else {
            assert(0 && "incorrect sharing mode");
        }

        if (x.if_zero) {
            XAMG_VECTOR_ALIGN
            for (uint64_t i = indx.r1 * NV; i < indx.r2 * NV; ++i)
                y_ptr[i] = 0.0;
        } else {
            for (uint64_t i = indx.r1; i < indx.r2; ++i) {
                uint64_t i1 = i * NV;
                uint64_t i2 = (i_ptr[i] - offset) * NV;

                XAMG_VECTOR_ALIGN
                for (uint16_t m = 0; m < NV; m++)
                    y_ptr[i1 + m] = x_ptr[i2 + m];
            }
        }
    }

    y.if_initialized = true;
    y.if_zero = x.if_zero;
}

//////////

template <typename T, const uint16_t NV>
inline void blas::forced_set_const(const vector::vector &x, const vector::vector &val) {

    vector::vector *xv = const_cast<vector::vector *>(&x);
    blas::set_const<T, NV>(*xv, val, true);
}

namespace blas {
#ifndef XAMG_SEPARATE_OBJECT
template <typename F>
std::map<int, vecs_t> ConstVectorsCache<F>::cache;
template struct ConstVectorsCache<float>;
template struct ConstVectorsCache<double>;
#endif
} // namespace blas

} // namespace XAMG
