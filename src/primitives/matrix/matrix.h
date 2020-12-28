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

#include "csr_matrix.h"
#include "dense_matrix.h"
#include "ell_matrix.h"

#include "matrix_chunk.h"

#include "part/part.h"
#include "misc/misc.h"

#include "detail/backend.inl"
#include "detail/stats.inl"

#include "comm/matvec_comm.h"

namespace XAMG {
namespace matrix {

enum segment_t { SEGMENT_UNDEFINED, SEGMENT_BY_BLOCKS, SEGMENT_BY_SLICES };

static inline mem::allocation get_vector_allocation_by_sharing(mem::sharing sharing_mode) {
    if (sharing_mode == mem::CORE)
        return mem::LOCAL;
    else if (sharing_mode == mem::NUMA)
        return mem::SHARED;

    assert(0);

    return mem::LOCAL;
}

struct matrix_block {
    bool reduced_prec = false;
    uint16_t hash = 0;
    uint32_t block_id = 0;

    mem::sharing sharing_mode;
    std::shared_ptr<backend> data = nullptr;
    std::shared_ptr<matrix_op_base> blas2_driver = nullptr;

    vector::vector inv_diag;
    vector::vector inv_sqrt_diag;

    matrix_block(mem::sharing sharing_mode_ = mem::CORE)
        : sharing_mode(sharing_mode_), inv_diag(sharing_mode_), inv_sqrt_diag(sharing_mode_) {}

    template <typename T>
    void assemble(const T &mat, const uint64_t &rnb) {
        uint16_t mat_hash = mat.encode_hash();
        if (reduced_prec) {
            mat_hash |= F_MASK;
            mat_hash &= (~F_MASK | F32_TYPE);
        }
        if (sharing_mode == mem::NUMA)
            mpi::bcast<uint16_t>(&mat_hash, 1, 0, mpi::INTRA_NUMA);

        std::shared_ptr<backend> backend_obj = create_matrix_block_obj(mat, mat_hash, sharing_mode);

        data = std::move(backend_obj);
        hash = mat_hash;
        block_id = rnb;
    }

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    void disassemble(csr_matrix<F, I1, I2, I3, I4> &mat, const uint32_t r1,
                     const uint32_t r2) const {
        assert(r1 >= 0);
        assert(r2 >= r1);
        assert(r2 <= data->get_nrows());

        //        size_t core_size = r2 - r1;
        //        size_t core_offset = r1;
        mat.nrows = r2 - r1;
        mat.nonzeros = data->get_range_size(r1, r2);
        mat.ncols = data->get_ncols();
        mat.block_nrows = r2 - r1;
        mat.block_ncols = data->get_block_ncols();
        mat.block_row_offset = data->get_block_row_offset() + r1;
        mat.block_col_offset = data->get_block_col_offset();
        mat.alloc();

        /////////

        std::vector<uint64_t> col;
        std::vector<float64_t> val;
        std::vector<F> val_F;

        for (uint64_t l = 0; l < mat.nrows; ++l) {
            data->get_row(l + r1, col, val);

            val_F.resize(val.size());
            for (size_t i = 0; i < val.size(); ++i)
                val_F[i] = (F)val[i];
            mat.upload_row(l, col, val_F, col.size());
        }
    }

    void get_inv_diag() {
        if (!inv_diag.if_allocated)
            data->alloc_inv_diag(inv_diag);
    }

    void get_inv_sqrt_diag() {
        if (!inv_sqrt_diag.if_allocated)
            data->alloc_inv_sqrt_diag(inv_sqrt_diag);
    }
};

static inline mem::sharing matrix_block_sharing_mode(segment::hierarchy layer,
                                                     mem::allocation alloc_mode) {
    if (layer == segment::CORE)
        return mem::CORE;

    if (alloc_mode == mem::LOCAL)
        return mem::CORE;
    if (alloc_mode == mem::DISTRIBUTED)
        return mem::NUMA;

    assert("the value must be specified" && 0);

    return mem::CORE;
}

////////////////////////////////////////

struct segmentation_layer {
    matrix_block diag;
    std::vector<matrix_block> offd;

    comm::comm_p2p p2p_comm;

    segmentation_layer(mem::sharing sharing_, segment::hierarchy layer_)
        : diag(sharing_), p2p_comm(layer_) {}
};

////////////////////////////////////////

struct matrix {

    bool reduced_prec;
    mem::allocation alloc_mode;
    segment_t segmentation;

    bool if_drivers_allocated;
    bool if_buffers_allocated;
    float64_t min_eig, max_eig;

    matrix_stats info;

    std::shared_ptr<part::part> row_part;
    std::shared_ptr<part::part> col_part;

    std::map<segment::hierarchy, segmentation_layer> data_layer;
    comm::comm_global global_comm;

    //    BK: placeholder for further experiments with SpMV and optimization of
    //    data transfers by local multiplication of some non-diag blocks
    //    std::vector<backend *> col_blocks;
    //    std::vector<blas2_base *> col_blas2_drivers;

    matrix(mem::allocation alloc_mode_ = mem::LOCAL)
        : reduced_prec(false), alloc_mode(alloc_mode_), segmentation(SEGMENT_UNDEFINED),
          if_drivers_allocated(false), if_buffers_allocated(false), min_eig(sigNaN),
          max_eig(sigNaN), row_part(nullptr), col_part(nullptr), global_comm(segment::NODE) {
        assert((alloc_mode == mem::LOCAL) || (alloc_mode == mem::DISTRIBUTED));
    }

    matrix(std::shared_ptr<part::part> _row_part, std::shared_ptr<part::part> _col_part,
           mem::allocation alloc_mode_)
        : reduced_prec(false), alloc_mode(alloc_mode_), segmentation(SEGMENT_UNDEFINED),
          if_drivers_allocated(false), if_buffers_allocated(false), min_eig(sigNaN),
          max_eig(sigNaN), row_part(_row_part), col_part(_col_part), global_comm(segment::NODE) {
        //        assert((alloc_mode == mem::LOCAL) || (alloc_mode == mem::DISTRIBUTED));
    }

    matrix(const matrix &that)
        : reduced_prec(that.reduced_prec), alloc_mode(that.alloc_mode),
          segmentation(that.segmentation), if_drivers_allocated(that.if_drivers_allocated),
          if_buffers_allocated(that.if_buffers_allocated), min_eig(that.min_eig),
          max_eig(that.max_eig), info(that.info), row_part(that.row_part), col_part(that.col_part),
          data_layer(that.data_layer), global_comm(that.global_comm) {}

    matrix &operator=(const matrix &that) {
        reduced_prec = that.reduced_prec;
        alloc_mode = that.alloc_mode;
        segmentation = that.segmentation;
        if_drivers_allocated = that.if_drivers_allocated;
        if_buffers_allocated = that.if_buffers_allocated;
        min_eig = that.min_eig;
        max_eig = that.max_eig;
        info = that.info;
        row_part = that.row_part;
        col_part = that.col_part;

        data_layer = that.data_layer;
        global_comm = that.global_comm;

        return *this;
    }

    ~matrix() {}

    /////////

    void add_data_layer(const segment::hierarchy layer);

    void set_part(const std::shared_ptr<part::part> part);
    void set_part(const std::shared_ptr<part::part> row_part,
                  const std::shared_ptr<part::part> col_part);
    void create_part(const uint64_t block_nrows);

    template <typename F, typename I>
    void segment(ell_matrix<F, I> &mat_ell, const uint32_t nblocks);

    //    template<typename F>
    //    void segment(dense_matrix<F> &mat_dense, const uint32_t nblocks);

    template <typename F, const uint16_t NV>
    void alloc_comm_buffers();

    const vector::vector &inv_diag() {
        auto layer = data_layer.find(segment::NUMA);
        assert(layer != data_layer.end());
        return layer->second.diag.inv_diag;
    }

    const vector::vector &inv_sqrt_diag() {
        auto layer = data_layer.find(segment::NUMA);
        assert(layer != data_layer.end());
        return layer->second.diag.inv_sqrt_diag;
    }

    /* out of date! */
    template <typename F, typename I>
    void assemble(const std::vector<ell_matrix<F, I>> &blocks,
                  const std::vector<uint64_t> &block_mapping_uncompress);
    /**/

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    void construct_diag_data(csr_matrix<F, I1, I2, I3, I4> &mat_csr,
                             const segment::hierarchy layer);

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    int construct_core_data(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks);

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    int construct_numa_data(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks);

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    int construct_node_data(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks);

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    void comm_send_objects(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks,
                           const segment::hierarchy layer);

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    void comm_recv_objects(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks,
                           const segment::hierarchy layer);

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    void construct(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks);

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    void construct(const csr_matrix<F, I1, I2, I3, I4> &csr_block);

    template <typename F>
    void construct_core_layer();

    void collect_matrix_stats();

    void print_comm_layer_stats(const segment::hierarchy hierarchy_layer,
                                const std::string comment);
    void print_comm_stats();

    std::vector<std::pair<uint64_t, uint64_t>>
    get_rows_range(const uint64_t &offset, const uint64_t &size,
                   const segment::hierarchy layer) const {
        uint64_t r1, r2;
        std::vector<std::pair<uint64_t, uint64_t>> range;
        const std::vector<matrix_block> &offd = data_layer.find(layer)->second.offd;

        for (uint32_t nb = 0; nb < offd.size(); ++nb) {
            if (offd[nb].data->if_empty) {
                range.emplace_back(std::make_pair(0, 0));
                continue;
            }
            assert(offd[nb].data->indexed());

            const auto &row_ind_ptr =
                offd[nb].data->get_row_ind_vector().get_aligned_ptr<uint32_t>();

            r1 = 0;
            for (uint32_t i = 0; i < offd[nb].data->nrows; ++i) {
                if (row_ind_ptr[i] >= offset) {
                    r1 = i;
                    break;
                }
            }

            r2 = offd[nb].data->nrows;
            for (uint32_t i = r1; i < offd[nb].data->nrows; ++i) {
                if (row_ind_ptr[i] >= offset + size) {
                    r2 = i;
                    break;
                }
            }

            range.emplace_back(std::make_pair(r1, r2));
        }
        return range;
    }
};

template <typename F, typename I1, typename I2, typename I3, typename I4>
std::shared_ptr<backend> create_matrix_block_obj(const csr_matrix<F, I1, I2, I3, I4> &mat_csr,
                                                 const uint16_t &hash,
                                                 mem::allocation = mem::LOCAL);

template <typename F>
std::shared_ptr<backend> create_matrix_block_obj(const dense_matrix<F> &mat_dense,
                                                 const uint16_t &hash);

template <typename F, typename I>
std::shared_ptr<backend> create_matrix_block_obj(const ell_matrix<F, I> &mat_ell,
                                                 const uint16_t &hash);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void compress_chunks(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks,
                     const segment::hierarchy layer);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void sync_chunks_data(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks,
                      const segment::hierarchy layer);

#ifndef XAMG_SEPARATE_OBJECT
matrix dummy(mem::LOCAL);
#else
extern matrix dummy;
#endif

struct matrix_container {

    bool if_value;
    bool if_reference;
    std::reference_wrapper<matrix> A_ref;
    matrix A_val;

    matrix_container(mem::allocation alloc_mode)
        : if_value(true), if_reference(false), A_ref(dummy), A_val(alloc_mode) {}
    matrix_container(matrix &A)
        : if_value(false), if_reference(true), A_ref(A), A_val(mem::LOCAL) {}

    void set_ref(matrix &A) {
        A_ref = A;
        if_value = false;
        if_reference = true;
    }

    void set_val(matrix &A) {
        assert(!if_reference && if_value);

        A_val = A;
    }

    matrix &get() {
        if (if_value)
            return A_val;
        else if (if_reference)
            return A_ref.get();
        else
            assert(0 && "attempt to access uninitialized matrix object");

        return *(new matrix); // never can be reached
    }
};

struct mg_layer {
    std::shared_ptr<part::part> part;

    matrix_container A;
    matrix_container R;
    matrix_container P;

    std::vector<vector::vector> buffer;

    mg_layer(mem::allocation alloc_mode) : A(alloc_mode), R(alloc_mode), P(alloc_mode) {}

    matrix &get_A() { return A.get(); }
    matrix &get_R() { return R.get(); }
    matrix &get_P() { return P.get(); }
};

template <typename M>
void construct_distributed(std::shared_ptr<XAMG::part::part> part, const M &local_mtx,
                           XAMG::matrix::matrix &distributed_mtx) {
    distributed_mtx.set_part(part);
    distributed_mtx.construct(local_mtx);
}

} // namespace matrix
} // namespace XAMG

///////////////////

#include "operations.h"

#include "detail/matrix.inl"
#include "detail/gen/matrix_block_obj.inl"
