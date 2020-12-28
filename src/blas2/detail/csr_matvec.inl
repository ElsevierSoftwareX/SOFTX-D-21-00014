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
void csr_Axpy(const matrix::csr_matrix<F, I1, I2, I3, I4> &m, const vector::vector &x,
              vector::vector &y, uint16_t nv) {
#ifdef XAMG_DBG_HEADER
    XAMG::out << XAMG::ALLRANKS << FUNC_PREFIX << "function , fp: <" << DEMANGLE_TYPEID_NAME(F)
              << ">; I1: <" << DEMANGLE_TYPEID_NAME(I1) << ">; I2: <" << DEMANGLE_TYPEID_NAME(I2)
              << ">; I3: <" << DEMANGLE_TYPEID_NAME(I3) << ">; I4: <" << DEMANGLE_TYPEID_NAME(I4)
              << ">\n";
#endif
    if (x.if_zero) {
        return;
    }
    if (m.if_empty) {
        y.if_zero = false;
        return;
    }
    assert(x.type_hash == y.type_hash);
    assert(m.val.type_hash == x.type_hash);
    m.row.check(vector::vector::initialized);
    m.col.check(vector::vector::initialized);
    m.val.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    y.check(vector::vector::initialized);

    const I1 *XAMG_RESTRICT row_ptr = m.row.template get_aligned_ptr<I1>();
    const I2 *XAMG_RESTRICT col_ptr = m.col.template get_aligned_ptr<I2>();
    const F *XAMG_RESTRICT val_ptr = m.val.template get_aligned_ptr<F>();

    const F *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<F>();
    int numa_block = -1;
    if (y.sharing_mode == mem::NODE)
        numa_block = id.nd_numa;
    F *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<F>(numa_block);

    uint64_t core_size, core_offset;
    uint64_t m_offset = 0;
    if ((x.sharing_mode == mem::CORE) && (y.sharing_mode == mem::NUMA_NODE)) {
        assert(m.sharing_mode == mem::CORE);
        core_size = m.nrows;
        core_offset = 0;
        m_offset = m.block_row_offset - y.global_numa_offset();
    } else if (y.sharing_mode == mem::NUMA_NODE) {
        if (m.if_indexed)
            y.get_core_range<I3, F>(core_size, core_offset, m.get_row_ind_vector());
        else
            y.get_core_range<F>(core_size, core_offset);
    } else {
        assert(0);
    }
    //    XAMG::out << ALLRANKS << "SpMV : AXPY " << core_offset << " : " << core_size << std::endl;

    /////////
    /*
        if (NV == 0) {
            assert(0);
            XAMG_STACK_ALIGNMENT_PREFIX F sum[nv];

            for (uint64_t i = core_offset; i < core_offset + core_size; i++) {
                for (uint16_t ii = 0; ii < nv; ii++)
                    sum[ii] = 0.0;

                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                    for (uint16_t ii = 0; ii < nv; ii++)
                        sum[ii] += x_ptr[col_ptr[j] * nv + ii] * val_ptr[j];
                }

                for (uint16_t ii = 0; ii < nv; ii++)
                    y_ptr[i * nv + ii] += sum[ii];
            }
        } else
    */
    {
        uint64_t row_shift, col_shift;

        if (!m.if_indexed) { // uncompressed matrix
            XAMG_VECTOR_ALIGN
            for (uint64_t i = core_offset; i < core_offset + core_size; i++) {
                row_shift = (i + m_offset) * NV;

                XAMG_NOVECTOR
                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                    col_shift = col_ptr[j] * NV;

                    XAMG_VECTOR_NODEP
                    XAMG_VECTOR_ALIGN
                    for (uint16_t ii = 0; ii < NV; ii++)
                        y_ptr[row_shift + ii] += x_ptr[col_shift + ii] * val_ptr[j];
                }
            }
        } else {
            m.row_ind.check(vector::vector::initialized);
            m.col_ind.check(vector::vector::initialized);
            const I3 *XAMG_RESTRICT row_ind_ptr = m.row_ind.template get_aligned_ptr<I3>();

            XAMG_VECTOR_ALIGN
            for (uint64_t i = core_offset; i < core_offset + core_size; i++) {
                row_shift = (row_ind_ptr[i] + m_offset) * NV;

                XAMG_NOVECTOR
                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                    col_shift = col_ptr[j] * NV;

                    XAMG_VECTOR_NODEP
                    XAMG_VECTOR_ALIGN
                    for (uint16_t ii = 0; ii < NV; ii++)
                        y_ptr[row_shift + ii] += x_ptr[col_shift + ii] * val_ptr[j];
                }
            }
        }
    }

    y.if_zero = false;
}

////////////////////

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
void csr_Ax_y(const matrix::csr_matrix<F, I1, I2, I3, I4> &m, const vector::vector &x,
              vector::vector &y, uint16_t nv) {
#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "function , fp: <" << DEMANGLE_TYPEID_NAME(F) << ">; I1: <"
              << DEMANGLE_TYPEID_NAME(I1) << ">; I2: <" << DEMANGLE_TYPEID_NAME(I2) << ">; I3: <"
              << DEMANGLE_TYPEID_NAME(I3) << ">; I4: <" << DEMANGLE_TYPEID_NAME(I4) << ">\n";
#endif
    if (x.if_zero) {
        blas::set_const<F, NV>(y, 0.0, true);
        return;
    }
    if (m.if_empty) {
        y.if_initialized = true;
        y.if_zero = false;
        return;
    }
    assert(x.type_hash == y.type_hash);
    assert(m.val.type_hash == x.type_hash);
    m.row.check(vector::vector::initialized);
    m.col.check(vector::vector::initialized);
    m.val.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    y.check(vector::vector::allocated);

    const I1 *XAMG_RESTRICT row_ptr = m.row.template get_aligned_ptr<I1>();
    const I2 *XAMG_RESTRICT col_ptr = m.col.template get_aligned_ptr<I2>();
    const F *XAMG_RESTRICT val_ptr = m.val.template get_aligned_ptr<F>();

    const F *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<F>();
    int numa_block = -1;
    if (y.sharing_mode == mem::NODE)
        numa_block = id.nd_numa;
    F *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<F>(numa_block);

    assert(y.sharing_mode == mem::NUMA_NODE);
    uint64_t core_size, core_offset;
    y.get_core_range<F>(core_size, core_offset);
    //    XAMG::out << ALLRANKS << m.nrows << " :: " << core_offset << " : " << core_size <<
    //    std::endl; XAMG::out << ALLRANKS << "SpMV : AX_Y " << core_offset << " : " << core_size <<
    //    std::endl;

    /////////
    /*
        if (NV == 0) {
            assert(0);
            XAMG_STACK_ALIGNMENT_PREFIX F sum[nv];

            for (uint64_t i = core_offset; i < core_offset + core_size; i++) {
                for (uint16_t ii = 0; ii < nv; ii++)
                    sum[ii] = 0.0;

                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                    for (uint16_t ii = 0; ii < nv; ii++)
                        sum[ii] += x_ptr[col_ptr[j] * nv + ii] * val_ptr[j];
                }

                for (uint16_t ii = 0; ii < nv; ii++)
                    y_ptr[i * nv + ii] = sum[ii];
            }
        } else
    */
    {
        uint64_t row_shift, col_shift;

        if (m.nrows == m.block_nrows) {
            XAMG_VECTOR_ALIGN
            for (uint64_t i = core_offset; i < core_offset + core_size; i++) {
                row_shift = i * NV;

                XAMG_VECTOR_ALIGN
                for (uint16_t ii = 0; ii < NV; ii++)
                    y_ptr[row_shift + ii] = 0.0;

                XAMG_NOVECTOR
                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                    col_shift = col_ptr[j] * NV;

                    XAMG_VECTOR_NODEP
                    XAMG_VECTOR_ALIGN
                    for (uint16_t ii = 0; ii < NV; ii++)
                        y_ptr[row_shift + ii] += x_ptr[col_shift + ii] * val_ptr[j];
                }
            }
        } else {
            m.row_ind.check(vector::vector::initialized);
            m.col_ind.check(vector::vector::initialized);
            const I3 *XAMG_RESTRICT row_ind_ptr = m.row_ind.template get_aligned_ptr<I3>();

            XAMG_VECTOR_ALIGN
            for (uint64_t i = core_offset; i < core_offset + core_size; i++) {
                row_shift = row_ind_ptr[i] * NV;

                XAMG_VECTOR_ALIGN
                for (uint16_t ii = 0; ii < NV; ii++)
                    y_ptr[row_shift + ii] = 0.0;

                XAMG_NOVECTOR
                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                    col_shift = col_ptr[j] * NV;

                    XAMG_VECTOR_NODEP
                    XAMG_VECTOR_ALIGN
                    for (uint16_t ii = 0; ii < NV; ii++)
                        y_ptr[row_shift + ii] += x_ptr[col_shift + ii] * val_ptr[j];
                }
            }
        }
    }

    y.if_initialized = true;
    y.if_zero = false;
}

} // namespace backend
} // namespace blas2
} // namespace XAMG
