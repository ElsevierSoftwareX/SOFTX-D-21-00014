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

#ifndef XAMG_SEPARATE_OBJECT
void matrix::set_part(const std::shared_ptr<part::part> _part) {
    row_part = _part;
    col_part = _part;
}

void matrix::set_part(const std::shared_ptr<part::part> _row_part,
                      const std::shared_ptr<part::part> _col_part) {
    row_part = _row_part;
    col_part = _col_part;
}

void matrix::create_part(const uint64_t block_size) {
    std::shared_ptr<part::part> part = part::get_shared_part();
    part->get_part(block_size);
    set_part(part);
}
#endif

///////////////////

template <typename F, const uint16_t NV>
void alloc_comm_layer_buffers(segmentation_layer &layer, std::shared_ptr<part::part> col_part) {
    assert(layer.p2p_comm.send.obj.size() == layer.p2p_comm.sender_indx.size());
    for (uint32_t i = 0; i < layer.p2p_comm.sender_indx.size(); ++i)
        layer.p2p_comm.send.obj[i].data.alloc<F>(layer.p2p_comm.sender_indx[i].size, NV);

    assert(layer.p2p_comm.recv.obj.size() == layer.offd.size());
    for (uint32_t i = 0; i < layer.p2p_comm.recv.obj.size(); ++i)
        layer.p2p_comm.recv.obj[i].data.alloc<F>(layer.offd[i].data->ncols, NV);

    for (auto &indx : layer.p2p_comm.sender_indx) {
        if (indx.sharing_mode == mem::CORE) {
            indx.r1 = 0;
            indx.r2 = indx.size;
        } else {
            // get range for each NUMA block and distribute it across the cores inside the numa block
            indx.set_part(col_part);
            uint32_t numa_block_size = 0;
            uint32_t numa_block_offset = 0;

            if (indx.sharing_mode == mem::NUMA) {
                numa_block_size = indx.size;
                numa_block_offset = 0;
            } else if (indx.sharing_mode == mem::NODE) {
                uint64_t offset = indx.vec_part->numa_layer.block_indx[id.nd_numa];
                uint64_t size = indx.vec_part->numa_layer.block_size[id.nd_numa];
                auto i_ptr = indx.get_aligned_ptr<uint32_t>();

                uint32_t r1 = 0;
                uint32_t r2 = 0;
                for (uint32_t i = 0; i < indx.size; ++i) {
                    if (i_ptr[i] < offset)
                        r1++;
                    if (i_ptr[i] < offset + size)
                        r2++;
                }

                numa_block_size = r2 - r1;
                numa_block_offset = r1;
            } else {
                assert(0);
            }

            uint32_t block_size = numa_block_size / id.nm_ncores;
            uint64_t block_offset = block_size * id.nm_core;
            if (id.nm_core == id.nm_ncores - 1)
                block_size = numa_block_size - block_offset;

            indx.r1 = block_offset + numa_block_offset;
            indx.r2 = indx.r1 + block_size;
        }
        //    }
        //            indx.r1 = 0;
        //            indx.r2 = indx.size;
        //
        //            uint64_t offset = indx.vec_part->core_layer.block_indx[id.nm_core];
        //            uint64_t size = indx.vec_part->core_layer.block_size[id.nm_core];
        //            auto i_ptr = indx.get_aligned_ptr<uint32_t>();
        //
        ////                TODO: optimize that
        //            for (uint64_t i = 0; i < indx.size; ++i) {
        //                if (i_ptr[i] < block_offset)
        //                    indx.r1++;
        //                if (i_ptr[i] < block_offset + block_size)
        //                    indx.r2++;
        //            }
        //        } else if (indx.sharing_mode == mem::NODE) {
        //            uint64_t offset = indx.vec_part->numa_layer.block_indx[id.nd_numa];
        //            auto i_ptr = indx.get_aligned_ptr<uint32_t>();
        //
        //            uint32_t numa_offset = 0;
        //            uint32_t numa_offset2 = 0;
        //
        //            for (uint64_t i = 0; i < indx.size; ++i) {
        //                if (i_ptr[i] < offset)
        //                    numa_offset++;
        //                if (i_ptr[i] < offset + indx.vec_part->numa_layer.block_size[id.nd_numa])
        //                    numa_offset2++;
        //            }
        //            uint32_t numa_block_size = numa_offset2 - numa_offset;
        //            uint32_t numa_block_indx = numa_offset;
        //        }
        //
        //        uint32_t block_size = numa_block_size / id.nm_ncores;
        //        uint64_t block_offset = block_size * id.nm_core;
        //        if (id.nm_core == id.nm_ncores-1)
        //            block_size = numa_block_size - block_offset;
        //
        //        indx.r1 = block_offset + numa_offset;
        //        indx.r2 = indx.r1 + block_size;
        //
        //            if (id.numa_master_process()) {
        //                uint64_t offset = indx.vec_part->numa_layer.block_indx[id.nd_numa];
        //                auto i_ptr = indx.get_aligned_ptr<uint32_t>();
        //
        ////                TODO: optimize that
        //                for (uint64_t i = 0; i < indx.size; ++i) {
        //                    if (i_ptr[i] < offset)
        //                        indx.r1++;
        //                    if (i_ptr[i] < offset +
        //                    indx.vec_part->numa_layer.block_size[id.nd_numa])
        //                        indx.r2++;
        //                }
        //            }
    }

    layer.p2p_comm.send_syncone.alloc();
    layer.p2p_comm.send_syncall.alloc();

    layer.p2p_comm.recv_syncone.alloc();
    layer.p2p_comm.recv_syncall.alloc();

    layer.p2p_comm.send_syncall2.alloc();

    layer.p2p_comm.numa_syncone.alloc();
    layer.p2p_comm.barrier.alloc();
    layer.p2p_comm.barrier2.alloc();

    layer.p2p_comm.send_syncall2.init();
    layer.p2p_comm.recv_syncone.init();
}

template <typename F, const uint16_t NV>
void matrix::alloc_comm_buffers() {
    if (segmentation == SEGMENT_BY_BLOCKS) {
        if (data_layer.find(segment::NODE) != data_layer.end())
            alloc_comm_layer_buffers<F, NV>(data_layer.find(segment::NODE)->second, col_part);
        else {
            assert(0);
        }

        if (data_layer.find(segment::NUMA) != data_layer.end())
            alloc_comm_layer_buffers<F, NV>(data_layer.find(segment::NUMA)->second, col_part);
        else {
            assert(0);
        }

        if (data_layer.find(segment::CORE) != data_layer.end())
            alloc_comm_layer_buffers<F, NV>(data_layer.find(segment::CORE)->second, col_part);
    } else if (segmentation == SEGMENT_BY_SLICES) {
        global_comm.alloc<F, NV>(col_part);
        global_comm.numa_syncone.alloc();
        global_comm.node_syncall.alloc();
        global_comm.node_syncone.alloc();
    }

    mpi::barrier(mpi::INTRA_NODE);
    if_buffers_allocated = true;
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
int matrix::construct_node_data(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks) {
    auto &node_layer = data_layer.find(segment::NODE)->second;

    node_layer.p2p_comm.create_send_objects(csr_chunks);
    sync_chunks_data(csr_chunks, segment::NODE);

    mem::sharing sharing_mode;
    if (alloc_mode == mem::LOCAL)
        sharing_mode = mem::CORE;
    else
        sharing_mode = mem::NUMA;

    ///////////////////

    int diag_block_id = -1;

    for (uint32_t nb = 0; nb < csr_chunks.size(); ++nb) {
        csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[nb]);

        if (chunk_wrapper.proc != (uint32_t)id.gl_node) {
            csr_matrix<F, I1, I2, I3, I4> segmented_block;

            if (id.numa_master_process())
                distribute(chunk_wrapper.mtx, segmented_block, row_part->numa_layer,
                           mpi::CROSS_NUMA);

            node_layer.offd.emplace_back(matrix_block(sharing_mode));
            node_layer.offd.back().reduced_prec = reduced_prec;
            node_layer.offd.back().assemble(segmented_block, chunk_wrapper.proc);
        } else {
            diag_block_id = nb;
        }
    }

    node_layer.p2p_comm.create_recv_objects(csr_chunks);

    return diag_block_id;
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
int matrix::construct_numa_data(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks) {
    auto &numa_layer = data_layer.find(segment::NUMA)->second;

    numa_layer.p2p_comm.create_send_objects(csr_chunks);
    sync_chunks_data(csr_chunks, segment::NUMA);

    mem::sharing sharing_mode;
    if (alloc_mode == mem::LOCAL)
        sharing_mode = mem::CORE;
    else
        sharing_mode = mem::NUMA;

    /////////

    int diag_block_id = -1;

    for (uint32_t nb = 0; nb < csr_chunks.size(); ++nb) {
        csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[nb]);

        if (chunk_wrapper.proc != (uint32_t)id.nd_numa) {
            numa_layer.offd.emplace_back(matrix_block(sharing_mode));
            numa_layer.offd.back().reduced_prec = reduced_prec;
            numa_layer.offd.back().assemble(chunk_wrapper.mtx, chunk_wrapper.proc);
        } else {
            diag_block_id = nb;
        }
    }

    numa_layer.p2p_comm.create_recv_objects(csr_chunks);

    return diag_block_id;
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
int matrix::construct_core_data(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks) {
    auto &core_layer = data_layer.find(segment::CORE)->second;

    core_layer.p2p_comm.create_send_objects(csr_chunks);

    /////////

    int diag_block_id = -1;

    for (uint32_t nb = 0; nb < csr_chunks.size(); ++nb) {
        csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[nb]);

        if (chunk_wrapper.proc != (uint32_t)id.nm_core) {
            core_layer.offd.emplace_back(matrix_block(mem::CORE));
            core_layer.offd.back().reduced_prec = reduced_prec;
            core_layer.offd.back().assemble(chunk_wrapper.mtx, chunk_wrapper.proc);
        } else {
            diag_block_id = nb;
        }
    }

    core_layer.p2p_comm.create_recv_objects(csr_chunks);

    return diag_block_id;
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void matrix::construct_diag_data(csr_matrix<F, I1, I2, I3, I4> &mat_csr,
                                 const segment::hierarchy layer) {
    int proc = 0;
    if (layer == segment::CORE) {
        proc = id.nm_core;
    } else if (layer == segment::NUMA) {
        proc = id.nd_numa;
    } else {
        assert(0);
    }

    auto &segmentation_layer = data_layer.find(layer)->second;
    segmentation_layer.diag.reduced_prec = reduced_prec;
    segmentation_layer.diag.assemble(mat_csr, proc);
}

#ifndef XAMG_SEPARATE_OBJECT
void matrix::add_data_layer(const segment::hierarchy layer) {
    data_layer.insert(std::make_pair(
        layer, segmentation_layer(matrix_block_sharing_mode(layer, alloc_mode), layer)));
}
#endif

template <typename F, typename I1, typename I2, typename I3, typename I4>
void matrix::construct(std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_node_chunks) {
    // XAMG::out << XAMG::ALLRANKS << "Constructing matrix..." << std::endl;
    csr_matrix<F, I1, I2, I3, I4> empty_csr;

    segmentation = SEGMENT_BY_BLOCKS;
    add_data_layer(segment::NODE);
    add_data_layer(segment::NUMA);

    int node_diag_block_id = construct_node_data(csr_node_chunks);

    if (node_diag_block_id != -1) {
        csr_matrix<F, I1, I2, I3, I4> csr_node_diag_block;
        std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> csr_numa_chunks;

        if (id.numa_master_process()) {
            csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(
                csr_node_chunks[node_diag_block_id]);
            distribute(chunk_wrapper.mtx, csr_node_diag_block, row_part->numa_layer,
                       mpi::CROSS_NUMA);

            split_by_columns(csr_node_diag_block, csr_numa_chunks, col_part->numa_layer);
            compress_chunks(csr_numa_chunks, segment::NUMA);
        }

        int numa_diag_block_id = construct_numa_data(csr_numa_chunks);

        if (numa_diag_block_id != -1) {
            csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(
                csr_numa_chunks[numa_diag_block_id]);
            construct_diag_data(chunk_wrapper.mtx, segment::NUMA);
        } else {
            construct_diag_data(empty_csr, segment::NUMA);
        }
    } else {
        construct_diag_data(empty_csr, segment::NUMA);
    }

    collect_matrix_stats();
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void matrix::construct(const csr_matrix<F, I1, I2, I3, I4> &csr_block) {
    csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t> csr_slice;
    collect(csr_block, csr_slice, mpi::INTRA_NODE);

    std::vector<csr_mtx_chunk_pair<F, I1, I2, I3, I4>> csr_chunks;
    if (id.node_master_process()) {
        split_by_columns(csr_slice, csr_chunks, col_part->node_layer);
        compress_chunks(csr_chunks, segment::NODE);
    }

    construct(csr_chunks);
}

template <typename F>
void matrix::construct_core_layer() {
    csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t> csr_slice;
    std::vector<csr_mtx_chunk_pair<F, uint32_t, uint32_t, uint32_t, uint32_t>> csr_core_chunks;
    csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t> empty_csr;

    if (data_layer.find(segment::CORE) != data_layer.end())
        return;

    add_data_layer(segment::CORE);
    auto &numa_layer = data_layer.find(segment::NUMA)->second;

    if (numa_layer.diag.data->if_empty) {
        construct_diag_data(empty_csr, segment::CORE);
        return;
    }

    uint32_t r1 = row_part->core_layer.block_indx[id.nm_core];
    uint32_t r2 = r1 + row_part->core_layer.block_size[id.nm_core];
    numa_layer.diag.disassemble(csr_slice, r1, r2);
    //  alternative:
    //    numa_layer.diag.disassemble(csr_slice);
    //    distribute(csr_slice, csr_block);

    split_by_columns(csr_slice, csr_core_chunks, col_part->core_layer);
    compress_chunks(csr_core_chunks, segment::CORE);

    int core_diag_block_id = construct_core_data(csr_core_chunks);

    if (core_diag_block_id != -1) {
        csr_mtx_chunk_wrapper<F, uint32_t, uint32_t, uint32_t, uint32_t> chunk_wrapper(
            csr_core_chunks[core_diag_block_id]);
        construct_diag_data(chunk_wrapper.mtx, segment::CORE);
    } else {
        construct_diag_data(empty_csr, segment::CORE);
    }

    //    core_diag.data->print();
}

#ifndef XAMG_SEPARATE_OBJECT
void matrix::print_comm_layer_stats(const segment::hierarchy hierarchy_layer,
                                    const std::string comment) {
    uint32_t comm_vol = 0;
    if (data_layer.find(hierarchy_layer) != data_layer.end()) {
        auto &layer = data_layer.find(hierarchy_layer)->second;
        for (auto &offd : layer.offd) {
            comm_vol += offd.data->ncols;
        }
        XAMG::out << XAMG::ALLRANKS << XAMG::LOG << "Proc " << id.gl_proc << " " << comment
                  << " neighbours: " << layer.offd.size() << " comm vol: " << comm_vol << std::endl;
    }
}

void matrix::print_comm_stats() {
    if (id.node_master_process() && (id.gl_nnodes > 1))
        print_comm_layer_stats(segment::NODE, "node");
    if (id.numa_master_process() && (id.nd_nnumas > 1))
        print_comm_layer_stats(segment::NUMA, "numa");
    //    print_comm_layer_stats(segment::CORE, "core");
    usleep(10000);
}

void matrix::collect_matrix_stats() {
    info.nrows = row_part->node_layer.block_indx[row_part->node_layer.nblocks - 1] +
                 row_part->node_layer.block_size[row_part->node_layer.nblocks - 1];
    info.ncols = col_part->node_layer.block_indx[col_part->node_layer.nblocks - 1] +
                 col_part->node_layer.block_size[col_part->node_layer.nblocks - 1];

    uint64_t local_nonzeros = 0;
    std::vector<uint32_t> local_nnz_per_row;
    uint32_t min_row_size = info.ncols;
    uint32_t max_row_size = 0;

    if (id.numa_master_process()) {
        auto &numa_layer = data_layer.find(segment::NUMA)->second;
        auto &node_layer = data_layer.find(segment::NODE)->second;

        uint32_t local_nrows = row_part->numa_layer.block_size[id.nd_numa];
        local_nnz_per_row.resize(local_nrows, 0);

        local_nonzeros = numa_layer.diag.data->get_nonzeros();
        for (size_t i = 0; i < numa_layer.diag.data->get_nrows(); ++i) {
            local_nnz_per_row[i] += numa_layer.diag.data->get_row_size(i);
        }

        for (uint32_t nb = 0; nb < numa_layer.offd.size(); ++nb) {
            const auto &data = numa_layer.offd[nb].data;
            local_nonzeros += data->get_nonzeros();

            const auto &row_indx = data->get_row_ind_vector();
            for (size_t i = 0; i < data->get_nrows(); ++i) {
                local_nnz_per_row[data->unpack_row_indx(i)] += data->get_row_size(i);
            }
        }

        for (uint32_t nb = 0; nb < node_layer.offd.size(); ++nb) {
            const auto &data = node_layer.offd[nb].data;
            local_nonzeros += data->get_nonzeros();

            const auto &row_indx = data->get_row_ind_vector();
            for (size_t i = 0; i < data->get_nrows(); ++i) {
                local_nnz_per_row[data->unpack_row_indx(i)] += data->get_row_size(i);
            }
        }
        for (uint32_t &elem : local_nnz_per_row) {
            min_row_size = std::min(min_row_size, elem);
            max_row_size = std::max(max_row_size, elem);
        }
    }

    mpi::allreduce_sum(&local_nonzeros, &(info.nonzeros), 1);
    mpi::allreduce_min(&min_row_size, &(info.nnz_per_row.min), 1);
    mpi::allreduce_max(&max_row_size, &(info.nnz_per_row.max), 1);
    info.nnz_per_row.avg = ((float64_t)info.nonzeros) / info.nrows;

    ////
    // comm nbrs
    if (id.node_master_process()) {
        const auto &node_layer = data_layer.find(segment::NODE)->second;
        uint32_t nbrs = node_layer.offd.size();
        uint32_t total;

        mpi::allreduce_min(&nbrs, &(info.comm_nbrs.min), 1, mpi::CROSS_NODE);
        mpi::allreduce_max(&nbrs, &(info.comm_nbrs.max), 1, mpi::CROSS_NODE);
        mpi::allreduce_sum(&nbrs, &total, 1, mpi::CROSS_NODE);
        info.comm_nbrs.avg = ((float64_t)total) / id.gl_nnodes;
    }
    mpi::bcast(&(info.comm_nbrs.min), 1, 0, mpi::INTRA_NODE);
    mpi::bcast(&(info.comm_nbrs.max), 1, 0, mpi::INTRA_NODE);
    mpi::bcast(&(info.comm_nbrs.avg), 1, 0, mpi::INTRA_NODE);

    ////
    // comm volume
    if (id.node_master_process()) {
        const auto &node_layer = data_layer.find(segment::NODE)->second;
        uint32_t proc_recv_vol = 0;
        for (const auto &offd : node_layer.offd) {
            proc_recv_vol += offd.data->get_ncols();
        }
        uint32_t total;

        mpi::allreduce_min(&proc_recv_vol, &(info.comm_volume.min), 1, mpi::CROSS_NODE);
        mpi::allreduce_max(&proc_recv_vol, &(info.comm_volume.max), 1, mpi::CROSS_NODE);
        mpi::allreduce_sum(&proc_recv_vol, &total, 1, mpi::CROSS_NODE);
        info.comm_volume.avg = ((float64_t)total) / id.gl_nnodes;
    }
    mpi::bcast(&(info.comm_volume.min), 1, 0, mpi::INTRA_NODE);
    mpi::bcast(&(info.comm_volume.max), 1, 0, mpi::INTRA_NODE);
    mpi::bcast(&(info.comm_volume.avg), 1, 0, mpi::INTRA_NODE);
}
#endif

} // namespace matrix
} // namespace XAMG
