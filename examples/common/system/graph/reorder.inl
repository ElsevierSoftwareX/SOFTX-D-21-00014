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

// TODO: local & global mapping functions are tested for NUMA/NODE layers, but need to be revised...

template <typename I>
void get_local_mapping(std::vector<I> &local_mapping, const std::vector<int> &partitioning,
                       const XAMG::segment::hierarchy &layer) {
    XAMG::mpi::comm_pool comm_group(XAMG::mpi::cross_layer_comm(layer));
    XAMG::mpi::token tok;

    std::vector<int> local_block_size(comm_group.nprocs, 0);
    std::vector<int> new_block_size(comm_group.nprocs, 0);
    for (size_t i = 0; i < partitioning.size(); ++i) {
        ++local_block_size[partitioning[i]];
    }

    XAMG::mpi::allreduce_sum<int>(local_block_size.data(), new_block_size.data(),
                                  local_block_size.size(), comm_group.comm);

    std::vector<int> new_block_indx(comm_group.nprocs, 0);

    // XAMG::out << XAMG::ALLRANKS << local_mapping.size() << " " << partitioning.size() <<
    // std::endl;

    if (!comm_group.master_proc) {
        XAMG::mpi::recv<int>(new_block_indx.data(), comm_group.nprocs, comm_group.proc - 1,
                             comm_group.comm, 0);
    } else {
        for (size_t i = 1; i < (size_t)comm_group.nprocs; ++i) {
            new_block_indx[i] = new_block_indx[i - 1] + new_block_size[i - 1];
        }
    }

    for (size_t i = 0; i < partitioning.size(); ++i) {
        uint32_t block = partitioning[i];
        local_mapping[i] = new_block_indx[block];
        ++new_block_indx[block];
    }

    if (comm_group.proc < comm_group.nprocs - 1) {
        XAMG::mpi::isend<int>(new_block_indx.data(), comm_group.nprocs, comm_group.proc + 1, tok,
                              comm_group.comm, 0);
        XAMG::mpi::wait(tok);
    }
}

template <typename I>
void get_global_mapping(std::vector<I> &global_mapping, std::vector<I> &local_mapping,
                        const XAMG::segment::hierarchy &layer) {
    const XAMG::mpi::scope cross_comm = XAMG::mpi::cross_layer_comm(layer);

    XAMG::part::part_layer node_layer, numa_layer, core_layer;

    if (layer == XAMG::segment::CORE) {
        assert(0);
        uint64_t core_block_size = local_mapping.size();
        core_layer.get_part_layer(core_block_size, XAMG::segment::CORE);

        if (id.numa_master_process()) {
            uint64_t numa_block_size = core_layer.block_indx.back();
            numa_layer.get_part_layer(numa_block_size, XAMG::segment::NUMA);

            if (id.node_master_process()) {
                uint64_t node_block_size = numa_layer.block_indx.back();
                node_layer.get_part_layer(node_block_size, XAMG::segment::NODE);
            }
        }
    } else if (layer == XAMG::segment::NUMA) {
        if (id.numa_master_process()) {
            uint64_t numa_block_size = local_mapping.size();
            numa_layer.get_part_layer(numa_block_size, XAMG::segment::NUMA);

            if (id.node_master_process()) {
                uint64_t node_block_size = numa_layer.block_indx.back();
                node_layer.get_part_layer(node_block_size, XAMG::segment::NODE);
            }
        }
    } else if (layer == XAMG::segment::NODE) {
        if (id.node_master_process()) {
            uint64_t node_block_size = local_mapping.size();
            node_layer.get_part_layer(node_block_size, XAMG::segment::NODE);
        }
    }

    ////

    std::vector<I> node_mapping;

    switch (layer) {
    case XAMG::segment::NUMA: {
        if (id.numa_master_process()) {
            std::vector<int> iblock_size(numa_layer.nblocks);
            std::vector<int> iblock_indx(numa_layer.nblocks);
            for (size_t i = 0; i < numa_layer.nblocks; ++i) {
                iblock_size[i] = numa_layer.block_size[i];
                iblock_indx[i] = numa_layer.block_indx[i];
            }

            if (id.node_master_process())
                node_mapping.resize(numa_layer.block_indx[numa_layer.nblocks]);

            XAMG::mpi::gatherv<I>(local_mapping.data(), local_mapping.size(), node_mapping.data(),
                                  iblock_size.data(), iblock_indx.data(), 0, XAMG::mpi::CROSS_NUMA);
        }

        ////
        if (id.node_master_process()) {
            std::vector<int> iblock_size(node_layer.nblocks);
            std::vector<int> iblock_indx(node_layer.nblocks);
            for (size_t i = 0; i < node_layer.nblocks; ++i) {
                iblock_size[i] = node_layer.block_size[i];
                iblock_indx[i] = node_layer.block_indx[i];
            }

            //            XAMG::out << XAMG::ALLRANKS << node_layer.block_indx[node_layer.nblocks]
            //            << global_mapping.size() << std::endl;
            //
            //            assert(node_layer.block_indx[node_layer.nblocks] ==
            //            global_mapping.size()); assert(iblock_size[id.gl_node] ==
            //            node_mapping.size());

            for (size_t i = 0; i < node_mapping.size(); ++i) {
                node_mapping[i] += iblock_indx[id.gl_node];
            }

            XAMG::mpi::allgatherv<I>(node_mapping.data(), node_mapping.size(),
                                     global_mapping.data(), iblock_size.data(), iblock_indx.data(),
                                     XAMG::mpi::CROSS_NODE);
        }

        break;
    }
    case XAMG::segment::NODE: {
        if (id.node_master_process()) {
            std::vector<int> iblock_size(node_layer.nblocks);
            std::vector<int> iblock_indx(node_layer.nblocks);
            for (size_t i = 0; i < node_layer.nblocks; ++i) {
                iblock_size[i] = node_layer.block_size[i];
                iblock_indx[i] = node_layer.block_indx[i];
            }

            XAMG::mpi::allgatherv<I>(local_mapping.data(), local_mapping.size(),
                                     global_mapping.data(), iblock_size.data(), iblock_indx.data(),
                                     XAMG::mpi::CROSS_NODE);
        }
        break;
    }
    default: {
        assert(0);
        break;
    }
    }

    //    part ...

    //    if (id.node_master_process())
    //        XAMG::mpi::allgatherv<I>(node_mapping.data(), node_mapping.size(),
    //        global_mapping.data(), node_layer.block_size.data(), node_layer.block_indx.data(),
    //        XAMG::mpi::CROSS_NODE);

    if (id.numa_master_process() && (layer != XAMG::segment::NODE))
        XAMG::mpi::bcast<I>(global_mapping.data(), global_mapping.size(), 0, XAMG::mpi::CROSS_NUMA);

    if (layer == XAMG::segment::CORE)
        XAMG::mpi::bcast<I>(global_mapping.data(), global_mapping.size(), 0, XAMG::mpi::INTRA_NUMA);
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void get_filtered_graph(const XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> &mat_csr,
                        XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> &filt_csr) {
    filt_csr.nrows = mat_csr.nrows;
    filt_csr.ncols = mat_csr.ncols;
    filt_csr.nonzeros = mat_csr.nonzeros - mat_csr.nrows;

    filt_csr.block_nrows = mat_csr.block_nrows;
    filt_csr.block_ncols = mat_csr.block_ncols;

    filt_csr.block_row_offset = mat_csr.block_row_offset;
    filt_csr.block_col_offset = mat_csr.block_col_offset;

    filt_csr.if_indexed = mat_csr.if_indexed;
    filt_csr.alloc();

    ////

    std::vector<uint64_t> col;
    std::vector<float64_t> val;
    std::vector<F> val_F;
    for (uint64_t l = 0; l < mat_csr.get_nrows(); ++l) {
        mat_csr.get_row(l, col, val);

        uint64_t ii = 0;
        while ((col[ii] + mat_csr.block_col_offset != l + mat_csr.block_row_offset) &&
               (ii < col.size())) {
            ++ii;
        }

        if (ii != col.size()) {
            col.erase(col.begin() + ii);
            val.erase(val.begin() + ii);
        }

        val_F.resize(val.size());
        for (size_t i = 0; i < val.size(); ++i)
            val_F[i] = (F)val[i];

        filt_csr.upload_row(l, col, val_F, col.size());
    }
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void get_graph_partitioning(const XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> &mat_csr,
                            std::vector<int> &partitioning, const XAMG::segment::hierarchy &layer) {
    XAMG::mpi::comm_pool comm_group(XAMG::mpi::cross_layer_comm(layer));

    XAMG::out << XAMG::LOG << "Graph reordering, layer " << layer << std::endl;
    XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> filt_csr;
    get_filtered_graph(mat_csr, filt_csr);

    XAMG::matrix::csr_matrix<F, int, int, int, int> graph_csr;
    XAMG::matrix::convert(filt_csr, graph_csr);
    //    graph_csr.print();

    ///////////////////
    //  vertex weights

    int local_block_size = graph_csr.nrows;
    std::vector<int> iproc_size(comm_group.nprocs, 0);
    XAMG::mpi::allgather<int>(&local_block_size, 1, iproc_size.data(), 1, comm_group.comm);

    std::vector<int> iproc_indx(comm_group.nprocs + 1);
    iproc_indx[0] = 0;
    for (size_t np = 0; np < (size_t)comm_group.nprocs; np++)
        iproc_indx[np + 1] = iproc_indx[np] + iproc_size[np];

    //    for (size_t np = 0; np < iproc_indx.size(); np++)
    //        XAMG::out << XAMG::ALLRANKS << iproc_indx[np] << std::endl;

    /////////

    partitioning.resize(graph_csr.nrows);

    auto row_ptr = graph_csr.row.template get_aligned_ptr<int>();
    auto col_ptr = graph_csr.col.template get_aligned_ptr<int>();

    std::vector<int> vwgt(graph_csr.nrows, 1);

    XAMG::mpi::barrier(comm_group.comm);
    double t1 = XAMG::sys::timer();

    graph_decomp(iproc_indx.data(), row_ptr, col_ptr, vwgt.data(), comm_group.nprocs,
                 partitioning.data(), layer);

    XAMG::mpi::barrier(comm_group.comm);
    double t2 = XAMG::sys::timer();

    XAMG::out << XAMG::LOG << "Decomposition completed : Matrix size = " << iproc_indx.back()
              << " \t Nblocks = " << comm_group.nprocs << " \t Partitioning time : " << t2 - t1
              << " sec." << std::endl;
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void data_mapping(std::shared_ptr<XAMG::matrix::csr_matrix<F, I1, I2, I3, I4>> &mat_csr,
                  std::vector<int> &partitioning, const XAMG::segment::hierarchy &layer) {

    std::vector<I2> local_mapping(partitioning.size());
    std::vector<I2> global_mapping(mat_csr->ncols);

    get_local_mapping<I2>(local_mapping, partitioning, layer);
    get_global_mapping<I2>(global_mapping, local_mapping, layer);

    mat_csr->permutation(global_mapping);
}
