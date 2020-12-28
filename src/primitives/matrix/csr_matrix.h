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

#include "detail/backend.inl"
#include "part/part.h"

#include "blas/blas.h"

#include "bit_operations.h"

#include "comm/data_exchange.h"

/////////

namespace XAMG {
namespace matrix {

template <typename F, typename I1, typename I2, typename I3, typename I4>
struct csr_matrix : backend {
    typedef F float_type;
    typedef I1 row_idx_type;
    typedef I2 col_idx_type;
    mem::sharing sharing_mode;

    vector::vector row;
    vector::vector col;
    vector::vector val;

    bool if_indexed;
    vector::vector row_ind;
    vector::vector col_ind;

    uint64_t nonzeros;

    csr_matrix(mem::sharing sharing_mode_ = mem::CORE)
        : sharing_mode(sharing_mode_), row(sharing_mode), col(sharing_mode), val(sharing_mode),
          if_indexed(false), row_ind(sharing_mode), col_ind(sharing_mode), nonzeros(0) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << XAMG::ALLRANKS << FUNC_PREFIX << "constructor, fp: <"
                  << DEMANGLE_TYPEID_NAME(F) << ">; I1: <" << DEMANGLE_TYPEID_NAME(I1) << ">; I2: <"
                  << DEMANGLE_TYPEID_NAME(I2) << ">; I3: <" << DEMANGLE_TYPEID_NAME(I3)
                  << ">; I4: <" << DEMANGLE_TYPEID_NAME(I4) << ">\n";
#endif
        assert((sharing_mode == mem::CORE) || (sharing_mode == mem::NUMA));

        if_empty = true;
        block_row_offset = block_col_offset = 0;
        block_nrows = block_ncols = 0;
        nrows = ncols = 0;

        row.set_type<I1>();
        col.set_type<I2>();
        val.set_type<F>();

        row_ind.set_type<I3>();
        col_ind.set_type<I4>();
    }

    virtual void Axpy(std::shared_ptr<matrix_op_base> b, const vector::vector &x, vector::vector &y,
                      uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << XAMG::ALLRANKS << FUNC_PREFIX << "function for CSR, fp: <"
                  << DEMANGLE_TYPEID_NAME(F) << ">; I1: <" << DEMANGLE_TYPEID_NAME(I1) << ">; I2: <"
                  << DEMANGLE_TYPEID_NAME(I2) << ">; I3: <" << DEMANGLE_TYPEID_NAME(I3)
                  << ">; I4: <" << DEMANGLE_TYPEID_NAME(I4) << ">\n";
#endif
        b->csr_Axpy(this, x, y, nv);
    }

    virtual void Ax_y(std::shared_ptr<matrix_op_base> b, const vector::vector &x, vector::vector &y,
                      uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << XAMG::ALLRANKS << FUNC_PREFIX << "function for CSR, fp: <"
                  << DEMANGLE_TYPEID_NAME(F) << ">; I1: <" << DEMANGLE_TYPEID_NAME(I1) << ">; I2: <"
                  << DEMANGLE_TYPEID_NAME(I2) << ">; I3: <" << DEMANGLE_TYPEID_NAME(I3)
                  << ">; I4: <" << DEMANGLE_TYPEID_NAME(I4) << ">\n";
#endif
        b->csr_Ax_y(this, x, y, nv);
    }

    virtual void SGS(std::shared_ptr<matrix_op_base> b, const vector::vector &inv_diag,
                     const vector::vector &t, vector::vector &x, const vector::vector &relax_factor,
                     uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "function for CSR, fp: <" << DEMANGLE_TYPEID_NAME(F)
                  << ">; I1: <" << DEMANGLE_TYPEID_NAME(I1) << ">; I2: <"
                  << DEMANGLE_TYPEID_NAME(I2) << ">; I3: <" << DEMANGLE_TYPEID_NAME(I3)
                  << ">; I4: <" << DEMANGLE_TYPEID_NAME(I4) << ">\n";
#endif
        if (!if_empty)
            b->csr_SGS(this, inv_diag, t, x, relax_factor, nv);
        else
            x.if_zero = t.if_zero;
    }

    bool alloc();
    // bool realloc();

    void compress();

    virtual bool indexed() const { return (if_indexed); }

    virtual const vector::vector &get_row_ind_vector() const {
        assert(if_indexed);

        return (row_ind);
    }

    virtual const vector::vector &get_col_ind_vector() const {
        assert(if_indexed);

        return (col_ind);
    }

    virtual uint64_t get_nonzeros() const { return (nonzeros); }

    virtual void get_row(const uint64_t l, std::vector<uint64_t> &_col,
                         std::vector<float32_t> &_val) const {
        assert(!if_empty);

        row.check(vector::vector::initialized);
        col.check(vector::vector::initialized);
        val.check(vector::vector::initialized);

        assert(l < nrows);
        uint64_t row_size = get_row_size(l);
        _col.resize(row_size, 0);
        _val.resize(row_size, 0.0);

        auto row_ptr = row.get_aligned_ptr<I1>();
        auto col_ptr = col.get_aligned_ptr<I2>();
        auto val_ptr = val.get_aligned_ptr<F>();

        for (uint64_t i = row_ptr[l]; i < (uint64_t)row_ptr[l + 1]; ++i) {
            _col[i - row_ptr[l]] = col_ptr[i];
            _val[i - row_ptr[l]] = val_ptr[i];
        }
    }

    virtual void get_row(const uint64_t l, std::vector<uint64_t> &_col,
                         std::vector<float64_t> &_val) const {
        assert(!if_empty);

        row.check(vector::vector::initialized);
        col.check(vector::vector::initialized);
        val.check(vector::vector::initialized);

        assert(l < nrows);
        uint64_t row_size = get_row_size(l);
        _col.resize(row_size, 0);
        _val.resize(row_size, 0.0);

        auto row_ptr = row.get_aligned_ptr<I1>();
        auto col_ptr = col.get_aligned_ptr<I2>();
        auto val_ptr = val.get_aligned_ptr<F>();

        for (I1 i = row_ptr[l]; i < row_ptr[l + 1]; ++i) {
            _col[i - row_ptr[l]] = col_ptr[i];
            _val[i - row_ptr[l]] = val_ptr[i];
        }
    }

    virtual void get_compressed_row(const uint64_t l, std::vector<uint64_t> &_col,
                                    std::vector<float32_t> &_val, uint32_t &indx) const {
        assert(0);
    }

    virtual void get_compressed_row(const uint64_t l, std::vector<uint64_t> &_col,
                                    std::vector<float64_t> &_val, uint32_t &indx) const {
        assert(!if_empty);
        assert(if_indexed);

        row.check(vector::vector::initialized);
        col.check(vector::vector::initialized);
        val.check(vector::vector::initialized);
        row_ind.check(vector::vector::initialized);

        auto row_ind_ptr = row_ind.get_aligned_ptr<I3>();

        while ((row_ind_ptr[indx] < (I3)l) && (indx < nrows)) {
            ++indx;
        }

        _col.resize(0);
        _val.resize(0);
        if (indx == nrows)
            return;

        if (row_ind_ptr[indx] == (I3)l) {
            uint64_t row_size = get_row_size(indx);
            _col.resize(row_size, 0);
            _val.resize(row_size, 0.0);

            auto row_ptr = row.get_aligned_ptr<I1>();
            auto col_ptr = col.get_aligned_ptr<I2>();
            auto val_ptr = val.get_aligned_ptr<F>();

            for (I1 i = row_ptr[indx]; i < row_ptr[indx + 1]; ++i) {
                _col[i - row_ptr[indx]] = col_ptr[i];
                _val[i - row_ptr[indx]] = val_ptr[i];
            }

            ++indx;
        }
    }

    virtual uint64_t get_range_size(const uint64_t l1, const uint64_t l2) const {
        assert(!if_empty);

        row.check(vector::vector::initialized);
        assert(l1 <= nrows);
        assert(l2 <= nrows);
        assert(l1 <= l2);

        auto row_ptr = row.get_aligned_ptr<I1>();
        return (row_ptr[l2] - row_ptr[l1]);
    };

    virtual uint64_t get_row_size(const uint64_t l) const { return (get_range_size(l, l + 1)); }

    virtual uint64_t unpack_row_indx(const uint64_t l) {
        assert(!if_empty);
        row_ind.check(vector::vector::initialized);
        auto row_ind_ptr = row_ind.get_aligned_ptr<I3>();
        return (row_ind_ptr[l]);
    }

    virtual uint64_t unpack_col_indx(const uint64_t l) {
        assert(!if_empty);
        col_ind.check(vector::vector::initialized);
        auto col_ind_ptr = col_ind.get_aligned_ptr<I4>();
        return (col_ind_ptr[l]);
    }

    virtual void print() const {
        if (if_empty)
            return;

        row.check(vector::vector::initialized);
        col.check(vector::vector::initialized);
        val.check(vector::vector::initialized);
        auto row_ptr = row.get_aligned_ptr<I1>();
        auto col_ptr = col.get_aligned_ptr<I2>();
        auto val_ptr = val.get_aligned_ptr<F>();

        XAMG::out << std::scientific << std::setprecision(3);
        if (!if_indexed) {
            for (uint64_t i = 0; i < nrows; i++) {
                XAMG::out << XAMG::ALLRANKS << "row " << i + block_row_offset << " : ";
                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++)
                    XAMG::out << "(" << (uint64_t)col_ptr[j] + block_col_offset << ":" << val_ptr[j]
                              << ") ";
                XAMG::out << std::endl;
            }
        } else {
            row_ind.check(vector::vector::initialized);
            col_ind.check(vector::vector::initialized);
            auto row_ind_ptr = row_ind.get_aligned_ptr<I3>();
            auto col_ind_ptr = col_ind.get_aligned_ptr<I4>();

            for (uint64_t i = 0; i < nrows; i++) {
                XAMG::out << XAMG::ALLRANKS << "row " << (uint64_t)row_ind_ptr[i] + block_row_offset
                          << " : ";
                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++)
                    XAMG::out << "(" << (uint64_t)col_ind_ptr[col_ptr[j]] + block_col_offset << ":"
                              << val_ptr[j] << ") ";
                XAMG::out << std::endl;
            }
        }
        XAMG::out << std::defaultfloat << std::setprecision(6);
    }

    virtual void alloc_inv_diag(vector::vector &inv_diag) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "constructor \n";
#endif
        if (if_empty) {
            inv_diag.alloc_and_get_aligned_ptr<F>(0);
            inv_diag.ext_offset = block_row_offset;
            inv_diag.if_zero = false;
            return;
        }

        row.check(vector::vector::initialized);
        col.check(vector::vector::initialized);
        val.check(vector::vector::initialized);

        assert(!if_indexed);
        auto row_ptr = row.get_aligned_ptr<I1>();
        auto col_ptr = col.get_aligned_ptr<I2>();
        auto val_ptr = val.get_aligned_ptr<F>();

        auto inv_diag_ptr = inv_diag.alloc_and_get_aligned_ptr<F>(nrows);

        for (uint64_t i = 0; i < nrows; ++i) {
            for (I1 j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                if ((I2)i == col_ptr[j])
                    inv_diag_ptr[i] = 1.0 / val_ptr[j];
            }
        }

        inv_diag.ext_offset = block_row_offset;
        inv_diag.if_initialized = true;
        inv_diag.if_zero = false;
    }

    virtual void alloc_inv_sqrt_diag(vector::vector &inv_sqrt_diag) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "constructor \n";
#endif
        if (if_empty) {
            inv_sqrt_diag.alloc_and_get_aligned_ptr<F>(0);
            inv_sqrt_diag.ext_offset = block_row_offset;
            inv_sqrt_diag.if_zero = false;
            return;
        }

        row.check(vector::vector::initialized);
        col.check(vector::vector::initialized);
        val.check(vector::vector::initialized);

        assert(!if_indexed);
        auto row_ptr = row.get_aligned_ptr<I1>();
        auto col_ptr = col.get_aligned_ptr<I2>();
        auto val_ptr = val.get_aligned_ptr<F>();

        auto inv_sqrt_diag_ptr = inv_sqrt_diag.alloc_and_get_aligned_ptr<F>(nrows);

        for (uint64_t i = 0; i < nrows; ++i) {
            for (I1 j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                if ((I2)i == col_ptr[j])
                    inv_sqrt_diag_ptr[i] = sqrt(1.0 / val_ptr[j]);
            }
        }

        inv_sqrt_diag.ext_offset = block_row_offset;
        inv_sqrt_diag.if_initialized = true;
        inv_sqrt_diag.if_zero = false;
    }

    void upload_row(const uint64_t nrow, const std::vector<uint64_t> &_col,
                    const std::vector<F> &_val, const uint32_t size) {
        // this function is valid only with sequential handling of matrix rows
        assert(!if_empty);
        assert(nrow < nrows);
        row.check(vector::vector::allocated);
        col.check(vector::vector::allocated);
        val.check(vector::vector::allocated);

        auto row_ptr = row.get_aligned_ptr<I1>();
        auto col_ptr = col.get_aligned_ptr<I2>();
        auto val_ptr = val.get_aligned_ptr<F>();

        if (!nrow)
            row_ptr[nrow] = 0;

        if (nrows) {
            I1 shift = row_ptr[nrow];
            for (uint64_t i = 0; i < size; ++i) {
                col_ptr[shift + i] = _col[i];
                val_ptr[shift + i] = (F)_val[i];
            }

            row_ptr[nrow + 1] = row_ptr[nrow] + size;
        }

        row.if_initialized = true;
        col.if_initialized = true;
        val.if_initialized = true;
        row.if_zero = false;
        col.if_zero = false;
        val.if_zero = false;
    }

    virtual uint16_t encode_hash() const {
        uint16_t hash = (define_int_type(nonzeros) << I1_OFFSET) |
                        (define_int_type(ncols) << I2_OFFSET) |
                        (define_int_type(block_nrows) << I3_OFFSET) |
                        (define_int_type(block_ncols) << I4_OFFSET);
        return (bit_encoding<F>() | hash);
    }

    void permutation(const std::vector<I2> &mapping) {
        assert(!if_indexed);
        assert(mapping.size() == ncols);
        col.check(vector::vector::initialized);
        auto col_ptr = col.get_aligned_ptr<I2>();

        for (uint64_t i = 0; i < nonzeros; ++i) {
            col_ptr[i] = mapping[col_ptr[i]];
        }
    }

    virtual void push_to_buffer(comm::data_exchange_buffer &buf) const override {
        buf.push_scalar<uint64_t>(block_row_offset);
        buf.push_scalar<uint64_t>(block_col_offset);
        buf.push_scalar<uint64_t>(block_nrows);
        buf.push_scalar<uint64_t>(block_ncols);
        buf.push_scalar<uint64_t>(nrows);
        buf.push_scalar<uint64_t>(ncols);

        buf.push_scalar<uint64_t>(nonzeros);

        assert(sharing_mode == mem::CORE);
        buf.push_scalar<bool>(if_indexed);

        if (!if_empty) {
            row.push_to_buffer<I1>(buf);
            col.push_to_buffer<I2>(buf);
            val.push_to_buffer<F>(buf);

            if (if_indexed) {
                row_ind.push_to_buffer<I3>(buf);
                col_ind.push_to_buffer<I4>(buf);
            }
        }
    }

    virtual void pull_from_buffer(comm::data_exchange_buffer &buf) override {
        buf.pull_scalar<uint64_t>(block_row_offset);
        buf.pull_scalar<uint64_t>(block_col_offset);
        buf.pull_scalar<uint64_t>(block_nrows);
        buf.pull_scalar<uint64_t>(block_ncols);
        buf.pull_scalar<uint64_t>(nrows);
        buf.pull_scalar<uint64_t>(ncols);

        buf.pull_scalar<uint64_t>(nonzeros);

        assert(sharing_mode == mem::CORE);
        buf.pull_scalar<bool>(if_indexed);

        alloc();

        if (!if_empty) {
            row.pull_from_buffer<I1>(buf);
            col.pull_from_buffer<I2>(buf);
            val.pull_from_buffer<F>(buf);

            if (if_indexed) {
                row_ind.pull_from_buffer<I3>(buf);
                col_ind.pull_from_buffer<I4>(buf);
            }
        }
    }

    void share_size_info() {
        mpi::bcast<uint64_t>(&nrows, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&ncols, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&nonzeros, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&block_nrows, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&block_ncols, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&block_row_offset, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&block_col_offset, 1, 0, mpi::INTRA_NUMA);
        uint8_t _ind = if_indexed;
        mpi::bcast<uint8_t>(&_ind, 1, 0, mpi::INTRA_NUMA);
        if_indexed = _ind;
    }
};

} // namespace matrix
} // namespace XAMG

#include "detail/csr_matrix.inl"
