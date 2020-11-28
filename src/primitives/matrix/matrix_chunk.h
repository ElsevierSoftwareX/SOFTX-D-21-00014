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

#include "csr_matrix.h"
#include "ell_matrix.h"
#include "dense_matrix.h"

namespace XAMG {
namespace matrix {

template <typename M>
using mtx_chunk_pair = std::pair<M, uint32_t>;

template <typename F, typename I1, typename I2, typename I3, typename I4>
using csr_mtx_chunk_pair = mtx_chunk_pair<csr_matrix<F, I1, I2, I3, I4>>;

template <typename F, typename I>
using ell_mtx_chunk_pair = mtx_chunk_pair<ell_matrix<F, I>>;

template <typename F>
using dense_mtx_chunk_pair = mtx_chunk_pair<dense_matrix<F>>;

template <typename M>
struct mtx_chunk_pair_wrapper {
    typename mtx_chunk_pair<M>::first_type &mtx;
    typename mtx_chunk_pair<M>::second_type &proc;

    mtx_chunk_pair_wrapper(mtx_chunk_pair<M> &p) : mtx(p.first), proc(p.second) {}
};

template <typename M>
struct const_mtx_chunk_pair_wrapper {
    const typename mtx_chunk_pair<M>::first_type &mtx;
    const typename mtx_chunk_pair<M>::second_type &proc;

    const_mtx_chunk_pair_wrapper(const mtx_chunk_pair<M> &p) : mtx(p.first), proc(p.second) {}
};

template <typename F, typename I1, typename I2, typename I3, typename I4>
using csr_mtx_chunk_wrapper = mtx_chunk_pair_wrapper<csr_matrix<F, I1, I2, I3, I4>>;

template <typename F, typename I1, typename I2, typename I3, typename I4>
using const_csr_mtx_chunk_wrapper = const_mtx_chunk_pair_wrapper<csr_matrix<F, I1, I2, I3, I4>>;

template <typename F, typename I1, typename I2, typename I3, typename I4>
void compress_chunks(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks,
                     const segment::hierarchy layer) {
    mpi::comm_pool comm_group(mpi::cross_layer_comm(layer));

    if (comm_group.member_proc) {
        for (uint32_t i = 0; i < csr_chunks.size(); ++i) {
            csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[i]);
            assert(!chunk_wrapper.mtx.if_empty);

            if (chunk_wrapper.proc != (uint32_t)comm_group.proc) {
                if (!chunk_wrapper.mtx.if_indexed)
                    chunk_wrapper.mtx.compress();
                else
                    XAMG::out << ALLRANKS << "Matrix is already compressed!" << std::endl;
            }
        }
    }
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void sync_chunks_data(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks,
                      const segment::hierarchy layer) {
    mpi::comm_pool cross_comm_group(mpi::cross_layer_comm(layer));
    mpi::comm_pool intra_comm_group(mpi::intra_layer_comm(layer));

    uint32_t nblocks = csr_chunks.size();
    mpi::bcast<uint32_t>(&nblocks, 1, 0, intra_comm_group.comm);
    if (!cross_comm_group.member_proc)
        csr_chunks.resize(nblocks);

    std::vector<uint32_t> block_proc(nblocks);
    if (cross_comm_group.member_proc) {
        for (uint32_t i = 0; i < csr_chunks.size(); ++i) {
            csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[i]);
            block_proc[i] = chunk_wrapper.proc;
        }
    }
    mpi::bcast<uint32_t>(block_proc.data(), nblocks, 0, intra_comm_group.comm);

    ////

    if (!cross_comm_group.member_proc) {
        for (uint32_t i = 0; i < csr_chunks.size(); ++i) {
            csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[i]);
            chunk_wrapper.proc = block_proc[i];
        }
    }
}

} // namespace matrix
} // namespace XAMG
