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

#include "../system/graph/scotch_wrapper.h"
#include "misc/reorder.h"

template <typename F, typename I1, typename I2, typename I3, typename I4>
void data_mapping(std::shared_ptr<XAMG::matrix::csr_matrix<F, I1, I2, I3, I4>> &mat,
                  std::vector<int> &partitioning, const XAMG::segment::hierarchy &layer);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void get_graph_partitioning(const XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> &mat,
                            std::vector<int> &partitioning, const XAMG::segment::hierarchy &layer);

template <typename F, typename I1, typename I2, typename I3, typename I4>
void redistribute_system(std::shared_ptr<XAMG::matrix::csr_matrix<F, I1, I2, I3, I4>> &mat,
                         std::shared_ptr<XAMG::vector::vector> &x,
                         std::shared_ptr<XAMG::vector::vector> &b,
                         const std::vector<int> &permutation,
                         const XAMG::segment::hierarchy &layer) {

    XAMG::map_info mapping_info(permutation, layer);

    XAMG::vector::redistribute<F>(x, mapping_info, layer);
    XAMG::vector::redistribute<F>(b, mapping_info, layer);
    XAMG::matrix::redistribute(mat, mapping_info, layer);
}

/////////////////////////////////////////////////////////////////////

template <typename F, typename I1, typename I2, typename I3, typename I4>
void reorder_system(std::shared_ptr<XAMG::matrix::csr_matrix<F, I1, I2, I3, I4>> &mat_csr,
                    std::shared_ptr<XAMG::vector::vector> &x,
                    std::shared_ptr<XAMG::vector::vector> &b) {
    XAMG::out << XAMG::LOG << "Graph reordering... " << std::endl;

    std::vector<int> partitioning;
    std::vector<I2> mapping;

    ////
    // NODE layer

    XAMG::vector::collect<F>(x, XAMG::mpi::INTRA_NODE);
    XAMG::vector::collect<F>(b, XAMG::mpi::INTRA_NODE);
    XAMG::matrix::collect(mat_csr, XAMG::mpi::INTRA_NODE);

    if (id.node_master_process()) {
        get_graph_partitioning(*mat_csr, partitioning, XAMG::segment::NODE);

        redistribute_system(mat_csr, x, b, partitioning, XAMG::segment::NODE);

        data_mapping(mat_csr, partitioning, XAMG::segment::NODE);
    }

    XAMG::part::part_layer node_part_layer;
    if (id.node_master_process())
        node_part_layer.get_part_layer(mat_csr->nrows, XAMG::segment::NODE);

    ////
    // NUMA layer

    if (id.numa_master_process()) {
        node_part_layer.share_part_layer(XAMG::segment::NUMA);

        XAMG::part::part_layer numa_part_layer;
        numa_part_layer.segment_part_layer(mat_csr->nrows, XAMG::segment::NUMA);

        XAMG::vector::distribute<F>(x, numa_part_layer, XAMG::mpi::CROSS_NUMA);
        XAMG::vector::distribute<F>(b, numa_part_layer, XAMG::mpi::CROSS_NUMA);
        XAMG::matrix::distribute(mat_csr, numa_part_layer, XAMG::mpi::CROSS_NUMA);
        ////
        if (id.nd_nnumas > 1) {
            // TODO: revise that, here we only want to take NUMA diag block of the matrix
            std::vector<XAMG::matrix::csr_mtx_chunk_pair<F, I1, I2, I3, I4>> csr_numa_chunks;
            XAMG::matrix::split_by_columns(*mat_csr, csr_numa_chunks, node_part_layer);
            uint32_t diag = 0;
            for (uint32_t i = 0; i < csr_numa_chunks.size(); ++i) {
                XAMG::matrix::csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(
                    csr_numa_chunks[i]);
                if (chunk_wrapper.proc == (uint32_t)id.gl_node)
                    diag = i;
            }
            XAMG::matrix::csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(
                csr_numa_chunks[diag]);
            ////

            get_graph_partitioning(chunk_wrapper.mtx, partitioning, XAMG::segment::NUMA);

            redistribute_system(mat_csr, x, b, partitioning, XAMG::segment::NUMA);

            data_mapping(mat_csr, partitioning, XAMG::segment::NUMA);
        }
    }

    XAMG::part::part_layer core_part_layer;
    core_part_layer.segment_part_layer(mat_csr->nrows, XAMG::segment::CORE);

    XAMG::vector::distribute<F>(x, core_part_layer, XAMG::mpi::INTRA_NUMA);
    XAMG::vector::distribute<F>(b, core_part_layer, XAMG::mpi::INTRA_NUMA);
    XAMG::matrix::distribute(mat_csr, core_part_layer, XAMG::mpi::INTRA_NUMA);

    //    b->print<F>("b");
    //    mat_csr->print();
    XAMG::mpi::barrier();
    XAMG::out << XAMG::LOG << "Graph reordering finished... " << std::endl;
    //    XAMG::io::sync();
}

#include "../system/graph/reorder.inl"
