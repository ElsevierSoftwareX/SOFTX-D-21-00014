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

template <typename F, typename I>
struct ell_matrix : backend {
    mem::sharing sharing_mode;

    vector::vector col;
    vector::vector val;

    uint16_t width;

    /////////

    ell_matrix(mem::sharing sharing_mode_)
        : sharing_mode(sharing_mode_), col(sharing_mode_), val(sharing_mode_), width(0) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "constructor, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">; I: <"
                  << DEMANGLE_TYPEID_NAME(I) << ">\n";
#endif
        col.set_type<I>();
        val.set_type<F>();

        block_row_offset = block_col_offset = 0;
        block_nrows = block_ncols = 0;
        nrows = ncols = 0;
        // io::print_bits(bit_hash);
    }

    template <typename F0, typename I01, typename I02, typename I03, typename I04>
    ell_matrix(csr_matrix<F0, I01, I02, I03, I04> &mat_csr)
        : sharing_mode(mem::CORE), col(sharing_mode), val(sharing_mode), width(0) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "constructor, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">; I: <"
                  << DEMANGLE_TYPEID_NAME(I) << ">\n";
#endif
        col.set_type<I>();
        val.set_type<F>();
        assert(mat_csr.if_indexed == false);

        block_row_offset = mat_csr.block_row_offset;
        block_col_offset = mat_csr.block_col_offset;

        block_nrows = mat_csr.block_nrows;
        block_ncols = mat_csr.block_ncols;

        nrows = mat_csr.nrows;
        ncols = mat_csr.ncols;

        // io::print_bits(bit_hash);
        /////////

        // width = 0;
        mat_csr.row.check(vector::vector::initialized);
        mat_csr.col.check(vector::vector::initialized);
        mat_csr.val.check(vector::vector::initialized);
        auto crow_ptr = mat_csr.row.template get_aligned_ptr<I01>();
        auto ccol_ptr = mat_csr.col.template get_aligned_ptr<I02>();
        auto cval_ptr = mat_csr.val.template get_aligned_ptr<F0>();

        for (uint64_t i = 0; i < mat_csr.nrows; ++i)
            width = std::max(width, (uint16_t)(crow_ptr[i + 1] - crow_ptr[i]));

        // XAMG::out << width << std::endl;
        alloc();

        col.check(vector::vector::allocated);
        val.check(vector::vector::allocated);

        auto ecol_ptr = col.get_aligned_ptr<I>();
        auto eval_ptr = val.get_aligned_ptr<F>();

        /////////

        for (uint64_t i = 0; i < nrows; ++i) {
            for (auto j = crow_ptr[i]; j < crow_ptr[i + 1]; ++j) {
                ecol_ptr[i * width + j - crow_ptr[i]] = ccol_ptr[j];
                eval_ptr[i * width + j - crow_ptr[i]] = cval_ptr[j];
            }

            for (auto j = crow_ptr[i + 1] - crow_ptr[i]; j < width; ++j) {
                ecol_ptr[i * width + j] = 0;
                eval_ptr[i * width + j] = 0.0;
            }
        }

        col.if_initialized = true;
        val.if_initialized = true;

        col.if_zero = false;
        val.if_zero = false;
    }

    ~ell_matrix(){};

    virtual void Axpy(std::shared_ptr<matrix_op_base> b, const vector::vector &x, vector::vector &y,
                      uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "function for ELL, fp: <" << DEMANGLE_TYPEID_NAME(F)
                  << ">; I: <" << DEMANGLE_TYPEID_NAME(I) << ">\n";
#endif
        b->ell_Axpy(this, x, y, nv);
    }

    virtual void Ax_y(std::shared_ptr<matrix_op_base> b, const vector::vector &x, vector::vector &y,
                      uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "function for ELL, fp: <" << DEMANGLE_TYPEID_NAME(F)
                  << ">; I: <" << DEMANGLE_TYPEID_NAME(I) << ">\n";
#endif
        b->ell_Ax_y(this, x, y, nv);
    }

    virtual void SGS(std::shared_ptr<matrix_op_base> b, const vector::vector &inv_diag,
                     const vector::vector &t, vector::vector &x, const vector::vector &relax_factor,
                     uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "function for ELL, fp: <" << DEMANGLE_TYPEID_NAME(F)
                  << ">; I: <" << DEMANGLE_TYPEID_NAME(I) << ">\n";
#endif
        b->ell_SGS(this, inv_diag, t, x, relax_factor, nv);
    }

    bool alloc() {
        uint64_t nonzeros = width * nrows;
        col.alloc<I>(nonzeros);
        val.alloc<F>(nonzeros);

        return true;
    }

    virtual bool indexed() const { return (false); }

    virtual const vector::vector &get_row_ind_vector() const {
        // FIXME!!!
        assert(0);
        return col;
        //        ...
    }

    virtual const vector::vector &get_col_ind_vector() const {
        // FIXME!!!
        assert(0);
        return col;
        //        ...
    }

    virtual uint64_t get_nonzeros() const { return (nrows * width); }

    //    virtual void get_row(const uint64_t l, std::vector<uint64_t> &col, std::vector<float32_t>
    //    &val) const {
    //        assert(0);
    //    }

    virtual void get_row(const uint64_t l, std::vector<uint64_t> &col,
                         std::vector<float64_t> &val) const {
        assert(0);
    }

    //    virtual void get_compressed_row(const uint64_t l, std::vector<uint64_t> &_col,
    //    std::vector<float32_t> &_val, uint32_t &indx) const {
    //        assert(0);
    //    }

    virtual void get_compressed_row(const uint64_t l, std::vector<uint64_t> &_col,
                                    std::vector<float64_t> &_val, uint32_t &indx) const {
        assert(0);
    }

    virtual uint64_t get_range_size(const uint64_t l1, const uint64_t l2) const {
        return (width * (l2 - l1));
    }

    virtual uint64_t get_row_size(const uint64_t l) const { return (width); }

    virtual uint64_t unpack_row_indx(const uint64_t l) {
        assert(0);
        return l;
    }

    virtual uint64_t unpack_col_indx(const uint64_t l) {
        assert(0);
        return l;
    }

    virtual void print() const {
        col.check(vector::vector::initialized);
        val.check(vector::vector::initialized);
        auto col_ptr = col.get_aligned_ptr<I>();
        auto val_ptr = val.get_aligned_ptr<F>();

        XAMG::out << std::scientific << std::setprecision(3);
        for (uint64_t i = 0; i < nrows; i++) {
            XAMG::out << XAMG::ALLRANKS << "row " << i + block_row_offset << " : ";
            for (uint16_t j = 0; j < width; ++j)
                XAMG::out << "(" << col_ptr[i * width + j] + block_col_offset << ":"
                          << val_ptr[i * width + j] << ") ";
            XAMG::out << std::endl;
        }
        XAMG::out << std::defaultfloat << std::setprecision(6);
    }

    virtual void alloc_inv_diag(vector::vector &inv_diag) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "constructor \n";
#endif
        col.check(vector::vector::initialized);
        val.check(vector::vector::initialized);
        auto col_ptr = col.get_aligned_ptr<I>();
        auto val_ptr = val.get_aligned_ptr<F>();

        auto inv_diag_ptr = inv_diag.alloc_and_get_aligned_ptr<F>(nrows);
        inv_diag.check(vector::vector::allocated);

        for (uint64_t i = 0; i < nrows; ++i) {
            for (uint64_t j = 0; j < width; ++j) {
                if (i == col_ptr[i * width + j]) {
                    inv_diag_ptr[i] = 1.0 / val_ptr[i * width + j];
                    break;
                }
            }
        }

        inv_diag.ext_offset = block_row_offset;
        inv_diag.if_initialized = true;
        inv_diag.if_zero = false;
    }

    virtual void alloc_inv_sqrt_diag(vector::vector &inv_sqrt_diag) { assert(0); }

    template <typename F0, typename I0>
    void fill_data(const ell_matrix<F0, I0> &block) {
        nrows = block.nrows;
        ncols = block.ncols;

        block_nrows = block.block_nrows;
        block_ncols = block.block_ncols;

        block_row_offset = block.block_row_offset;
        block_col_offset = block.block_col_offset;
        width = block.width;

        alloc();

        /////////
        if (!if_empty) {
            vector::convert<I0, I>(block.col, col);
            vector::convert<F0, F>(block.val, val);
        }
    }

    virtual uint16_t encode_hash() const {
        uint16_t hash = (define_int_type(ncols) << I1_OFFSET);
        return (bit_encoding<F>() | hash);
    }

    virtual void push_to_buffer(comm::data_exchange_buffer &buf) const override {
        buf.push_scalar<uint64_t>(block_row_offset);
        buf.push_scalar<uint64_t>(block_col_offset);
        buf.push_scalar<uint64_t>(block_nrows);
        buf.push_scalar<uint64_t>(block_ncols);
        buf.push_scalar<uint64_t>(nrows);
        buf.push_scalar<uint64_t>(ncols);

        assert(0);

        //        buf.push_scalar<uint64_t>(mat.nonzeros);
        //
        //        buf.push_scalar<bool>(mat.if_indexed);
        //
        //        push_vector<I1>(mat.row);
        //        push_vector<I2>(mat.col);
        //        push_vector<F>(mat.val);
        //
        //        if (mat.if_indexed) {
        //            push_vector<I3>(mat.row_ind);
        //            push_vector<I4>(mat.col_ind);
        //        }
    }

    virtual void pull_from_buffer(comm::data_exchange_buffer &buf) override {
        buf.pull_scalar<uint64_t>(block_row_offset);
        buf.pull_scalar<uint64_t>(block_col_offset);
        buf.pull_scalar<uint64_t>(block_nrows);
        buf.pull_scalar<uint64_t>(block_ncols);
        buf.pull_scalar<uint64_t>(nrows);
        buf.pull_scalar<uint64_t>(ncols);

        assert(0);
        //        pull_scalar<uint64_t>(mat.nonzeros);
        //
        //        pull_scalar<bool>(mat.if_indexed);
        //
        //        mat.alloc();
        //
        //        pull_vector<I1>(mat.row);
        //        pull_vector<I2>(mat.col);
        //        pull_vector<F>(mat.val);
        //
        //        if (mat.if_indexed) {
        //            pull_vector<I3>(mat.row_ind);
        //            pull_vector<I4>(mat.col_ind);
        //        }
    }
};

} // namespace matrix
} // namespace XAMG
