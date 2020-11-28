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
namespace matrix {

struct matrix_op_base {
    virtual void csr_Axpy(void *p, const XAMG::vector::vector &x, XAMG::vector::vector &y,
                          uint16_t nv) = 0;
    virtual void csr_Ax_y(void *p, const XAMG::vector::vector &x, XAMG::vector::vector &y,
                          uint16_t nv) = 0;
    virtual void ell_Axpy(void *p, const XAMG::vector::vector &x, XAMG::vector::vector &y,
                          uint16_t nv) = 0;
    virtual void ell_Ax_y(void *p, const XAMG::vector::vector &x, XAMG::vector::vector &y,
                          uint16_t nv) = 0;
    virtual void dense_Axpy(void *p, const XAMG::vector::vector &x, XAMG::vector::vector &y,
                            uint16_t nv) = 0;
    virtual void dense_Ax_y(void *p, const XAMG::vector::vector &x, XAMG::vector::vector &y,
                            uint16_t nv) = 0;

    virtual void csr_SGS(void *p, const XAMG::vector::vector &inv_diag,
                         const XAMG::vector::vector &t, XAMG::vector::vector &x,
                         const vector::vector &relax_factor, uint16_t nv) = 0;
    virtual void ell_SGS(void *p, const XAMG::vector::vector &inv_diag,
                         const XAMG::vector::vector &t, XAMG::vector::vector &x,
                         const vector::vector &relax_factor, uint16_t nv) = 0;
    virtual void dense_SGS(void *p, const XAMG::vector::vector &inv_diag,
                           const XAMG::vector::vector &t, XAMG::vector::vector &x,
                           const vector::vector &relax_factor, uint16_t nv) = 0;
};

struct backend {
    bool if_empty;

    uint64_t block_row_offset; // offset for the corresponding block in the global matrix
    uint64_t block_col_offset; // offset for the corresponding block in the global matrix

    uint64_t block_nrows, block_ncols; // uncompressed local block size
    uint64_t nrows, ncols; // compressed local block size after after filtering empty rows & cols

    /////////

    virtual ~backend(){};

    virtual void Axpy(std::shared_ptr<matrix_op_base> b, const vector::vector &x, vector::vector &y,
                      uint16_t nv = 0) {}
    virtual void Ax_y(std::shared_ptr<matrix_op_base> b, const vector::vector &x, vector::vector &y,
                      uint16_t nv = 0) {}

    virtual void SGS(std::shared_ptr<matrix_op_base> b, const vector::vector &inv_diag,
                     const vector::vector &t, vector::vector &x, const vector::vector &relax_factor,
                     uint16_t nv = 0) {}

    virtual void print() const = 0;

    virtual bool indexed() const = 0;

    virtual const vector::vector &get_row_ind_vector() const = 0;
    virtual const vector::vector &get_col_ind_vector() const = 0;

    virtual uint64_t get_nonzeros() const = 0;

    uint64_t get_nrows() const { return (nrows); }

    uint64_t get_ncols() const { return (ncols); }

    uint64_t get_block_nrows() const { return (block_nrows); }

    uint64_t get_block_ncols() const { return (block_ncols); }

    uint64_t get_block_row_offset() const { return (block_row_offset); }

    uint64_t get_block_col_offset() const { return (block_col_offset); }

    virtual void get_row(const uint64_t l, std::vector<uint64_t> &col,
                         std::vector<float64_t> &val) const = 0;

    virtual void get_compressed_row(const uint64_t l, std::vector<uint64_t> &col,
                                    std::vector<float64_t> &val, uint32_t &indx) const = 0;

    virtual uint64_t get_range_size(const uint64_t l1, const uint64_t l2) const = 0;
    virtual uint64_t get_row_size(const uint64_t l) const = 0;
    virtual uint64_t unpack_row_indx(const uint64_t l) = 0;
    virtual uint64_t unpack_col_indx(const uint64_t l) = 0;

    virtual void alloc_inv_diag(vector::vector &vec) = 0;
    virtual void alloc_inv_sqrt_diag(vector::vector &vec) = 0;

    void get_row(const uint64_t l, std::vector<int> &col, std::vector<float64_t> &val) const {
        std::vector<uint64_t> col_u64(0);
        get_row(l, col_u64, val);

        col.resize(col_u64.size());
        for (uint64_t i = 0; i < col_u64.size(); ++i) {
            assert(col_u64[i] < INT_MAX);
            col[i] = (int)col_u64[i];
        }
    }

    void get_row_and_unpack(const uint64_t l, std::vector<int> &col, std::vector<float64_t> &val,
                            uint32_t &indx) {
        std::vector<uint64_t> col_u64(0);
        uint64_t col_offset = get_block_col_offset();

        get_compressed_row(l, col_u64, val, indx);

        col.resize(col_u64.size());
        for (uint64_t i = 0; i < col_u64.size(); ++i) {
            assert(col_u64[i] < INT_MAX);
            col[i] = unpack_col_indx(col_u64[i]) + col_offset;
        }
    }

    virtual uint16_t encode_hash() const = 0;
    virtual void push_to_buffer(comm::data_exchange_buffer &buf) const = 0;
    virtual void pull_from_buffer(comm::data_exchange_buffer &buf) = 0;
};

} // namespace matrix
} // namespace XAMG
