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

template <typename F>
struct dense_matrix : backend {
    mem::sharing sharing_mode;

    vector::vector val;

    bool if_indexed;
    vector::vector row_ind;
    vector::vector col_ind;

    /////////

    dense_matrix(mem::sharing sharing_mode_)
        : sharing_mode(sharing_mode_), val(sharing_mode), if_indexed(false), row_ind(sharing_mode),
          col_ind(sharing_mode) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "constructor, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
        assert(sharing_mode == mem::CORE);
        val.set_type<F>();
        row_ind.set_type<uint32_t>();
        col_ind.set_type<uint32_t>();

        if_empty = true;
        block_row_offset = block_col_offset = 0;
        block_nrows = block_ncols = 0;
        nrows = ncols = 0;
    }

    template <typename F0, typename I01, typename I02, typename I03, typename I04>
    dense_matrix(csr_matrix<F0, I01, I02, I03, I04> &mat_csr)
        : val(mem::CORE), if_indexed(false), row_ind(mem::CORE), col_ind(mem::CORE) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "constructor, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
        val.set_type<F>();
        row_ind.set_type<uint32_t>();
        col_ind.set_type<uint32_t>();
        assert(!mat_csr.indexed());

        block_row_offset = mat_csr.get_block_row_offset();
        block_col_offset = mat_csr.get_block_col_offset();
        block_nrows = mat_csr.get_block_nrows();
        block_ncols = mat_csr.get_block_ncols();

        nrows = mat_csr.get_nrows();
        ncols = mat_csr.get_ncols();
        if_empty = mat_csr.if_empty;
        if (if_empty)
            return;

        alloc();
        blas::set_const<F, 1>(val, 0.0, true);

        val.check(vector::vector::initialized);
        auto dval_ptr = val.get_aligned_ptr<F>();

        mat_csr.row.check(vector::vector::initialized);
        mat_csr.col.check(vector::vector::initialized);
        mat_csr.val.check(vector::vector::initialized);
        auto row_ptr = mat_csr.row.template get_aligned_ptr<I01>();
        auto col_ptr = mat_csr.col.template get_aligned_ptr<I02>();
        auto val_ptr = mat_csr.val.template get_aligned_ptr<F0>();

        if (!mat_csr.indexed()) {
            for (uint64_t i = 0; i < nrows; ++i) {
                uint64_t irow = i; // + mat_csr.get_block_row_offset();

                for (auto j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                    uint64_t icol = col_ptr[j]; // + mat_csr.get_block_col_offset();

                    dval_ptr[irow * ncols + icol] = val_ptr[j];
                }
            }
        } else {
            assert(0);
        }

        val.if_initialized = true;
        val.if_zero = false;
    }

    dense_matrix(std::shared_ptr<backend> mat)
        : val(mem::CORE), if_indexed(false), row_ind(mem::CORE), col_ind(mem::CORE) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "constructor, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
        val.set_type<F>();
        row_ind.set_type<uint32_t>();
        col_ind.set_type<uint32_t>();

        if_indexed = false;
        block_row_offset = mat->get_block_row_offset();
        block_col_offset = mat->get_block_col_offset();
        block_nrows = mat->get_block_nrows();
        block_ncols = mat->get_block_ncols();

        nrows = mat->get_block_nrows(); // dense matrix is uncompressed!
        ncols = mat->get_block_ncols();

        if_empty = mat->if_empty;
        if (if_empty)
            return;

        alloc();
        blas::set_const<F, 1>(val, 0.0, true);

        val.check(vector::vector::initialized);
        auto val_ptr = val.get_aligned_ptr<F>();
        std::vector<uint64_t> col_b;
        std::vector<float64_t> val_b;

        if (!mat->indexed()) {
            for (uint64_t i = 0; i < mat->nrows; ++i) {
                mat->get_row(i, col_b, val_b);

                for (uint64_t j = 0; j < val_b.size(); ++j) {
                    val_ptr[i * ncols + col_b[j]] = (F)val_b[j];
                }
            }
        } else {
            for (uint64_t i = 0; i < mat->nrows; ++i) {
                mat->get_row(i, col_b, val_b);

                auto ii = mat->unpack_row_indx(i);
                for (uint64_t j = 0; j < val_b.size(); ++j) {
                    auto jj = mat->unpack_col_indx(col_b[j]);
                    val_ptr[ii * ncols + jj] = (F)val_b[j];
                }
            }
        }
        val.if_initialized = true;
        val.if_zero = false;
    }

    ~dense_matrix(){};

    ///////////////////

    virtual void Axpy(std::shared_ptr<matrix_op_base> b, const vector::vector &x, vector::vector &y,
                      uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "function for DENSE, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
        b->dense_Axpy(this, x, y, nv);
    }

    virtual void Ax_y(std::shared_ptr<matrix_op_base> b, const vector::vector &x, vector::vector &y,
                      uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "function for DENSE, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
        b->dense_Ax_y(this, x, y, nv);
    }

    virtual void SGS(std::shared_ptr<matrix_op_base> b, const vector::vector &inv_diag,
                     const vector::vector &t, vector::vector &x, const vector::vector &relax_factor,
                     uint16_t nv) {
#ifdef XAMG_DBG_HEADER
        XAMG::out << FUNC_PREFIX << "function for DENSE, fp: <" << DEMANGLE_TYPEID_NAME(F) << ">\n";
#endif
        b->dense_SGS(this, inv_diag, t, x, relax_factor, nv);
    }

    bool alloc();

    virtual bool indexed() const { return (if_indexed); }

    virtual const vector::vector &get_row_ind_vector() const {
        assert(if_indexed);

        return (row_ind);
    }

    virtual const vector::vector &get_col_ind_vector() const {
        assert(if_indexed);

        return (col_ind);
    }

    virtual uint64_t get_nonzeros() const { return (nrows * ncols); }

    virtual void get_row(const uint64_t l, std::vector<uint64_t> &_col,
                         std::vector<float64_t> &_val) const {
        val.check(vector::vector::initialized);

        uint64_t row_size = get_row_size(l);
        assert(row_size == ncols);
        _col.resize(row_size, 0);
        _val.resize(row_size, 0.0);

        auto val_ptr = val.get_aligned_ptr<F>();

        for (uint64_t i = 0; i < ncols; ++i) {
            _col[i] = i;
            _val[i] = (float64_t)val_ptr[l * ncols + i];
        }
    }

    virtual void get_compressed_row(const uint64_t l, std::vector<uint64_t> &_col,
                                    std::vector<float64_t> &_val, uint32_t &indx) const {
        assert(0);
    }

    virtual uint64_t get_row_size(const uint64_t l) const { return (ncols); }

    virtual uint64_t get_range_size(const uint64_t l1, const uint64_t l2) const {
        return (ncols * (l2 - l1));
    }

    virtual uint64_t unpack_row_indx(const uint64_t l) {
        row_ind.check(vector::vector::initialized);
        auto row_ind_ptr = row_ind.get_aligned_ptr<uint32_t>();
        return (row_ind_ptr[l]);
    }

    virtual uint64_t unpack_col_indx(const uint64_t l) {
        col_ind.check(vector::vector::initialized);
        auto col_ind_ptr = col_ind.get_aligned_ptr<uint32_t>();
        return (col_ind_ptr[l]);
    }

    virtual void print() const {
        if (if_empty)
            return;
        val.check(vector::vector::initialized);
        auto val_ptr = val.get_aligned_ptr<F>();

        XAMG::out << std::scientific << std::setprecision(3);
        for (uint64_t i = 0; i < nrows; i++) {
            XAMG::out << XAMG::ALLRANKS << "row " << i + block_row_offset << " : ";
            for (uint64_t j = 0; j < ncols; ++j)
                XAMG::out << val_ptr[i * ncols + j] << " ";
            XAMG::out << std::endl;
        }
        XAMG::out << std::defaultfloat << std::setprecision(6);
    }

    virtual void alloc_inv_diag(vector::vector &inv_diag) {
        val.check(vector::vector::initialized);
        auto val_ptr = val.get_aligned_ptr<F>();

        auto inv_diag_ptr = inv_diag.alloc_and_get_aligned_ptr<F>(nrows);
        inv_diag.check(vector::vector::allocated);

        for (uint64_t i = 0; i < nrows; ++i)
            inv_diag_ptr[i] = 1.0 / val_ptr[i * ncols + i];

        inv_diag.if_initialized = true;
        inv_diag.if_zero = false;
    }

    virtual void alloc_inv_sqrt_diag(vector::vector &inv_sqrt_diag) {
        val.check(vector::vector::initialized);
        assert(0);
    }

    void compress();

    void upload_row(const uint64_t nrow, const std::vector<float64_t> &_val, const uint32_t size) {
        assert(!if_empty);
        assert(nrow < nrows);
        val.check(vector::vector::allocated);

        auto val_ptr = val.get_aligned_ptr<F>();

        if (nrows) {
            uint64_t shift = nrow * ncols;
            for (uint64_t i = 0; i < size; ++i)
                val_ptr[shift + i] = (F)_val[i];
        }

        val.if_initialized = true;
        val.if_zero = false;
    }

    virtual uint16_t encode_hash() const { return (bit_encoding<F>()); }

    virtual void push_to_buffer(comm::data_exchange_buffer &buf) const override {
        buf.push_scalar<uint64_t>(block_row_offset);
        buf.push_scalar<uint64_t>(block_col_offset);
        buf.push_scalar<uint64_t>(block_nrows);
        buf.push_scalar<uint64_t>(block_ncols);
        buf.push_scalar<uint64_t>(nrows);
        buf.push_scalar<uint64_t>(ncols);

        // assert(alloc_mode == mem::LOCAL);
        buf.push_scalar<bool>(if_indexed);

        if (!if_empty) {
            val.push_to_buffer<F>(buf);

            if (if_indexed) {
                row_ind.push_to_buffer<uint32_t>(buf);
                col_ind.push_to_buffer<uint32_t>(buf);
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

        buf.pull_scalar<bool>(if_indexed);

        alloc();

        if (!if_empty) {
            val.pull_from_buffer<F>(buf);

            if (if_indexed) {
                row_ind.pull_from_buffer<uint32_t>(buf);
                col_ind.pull_from_buffer<uint32_t>(buf);
            }
        }
    }
};

} // namespace matrix
} // namespace XAMG

#include "detail/dense_matrix.inl"
