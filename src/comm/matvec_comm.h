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
#include "primitives/matrix/matrix_chunk.h"

#include "comm/mpi_token.h"
#include "comm/mpi_wrapper.h"

#include "comm/shm_sync.h"

#include "blas/blas.h"

///////////////////

namespace XAMG {
namespace comm {

struct matvec_comm_data {
    vector::vector data;
    uint32_t nbr;

    matvec_comm_data(mem::sharing sharing_mode) : data(sharing_mode), nbr(0) {}

    template <typename T, const uint16_t NV>
    inline void mpi_recv_anysize(mpi::scope comm) {
        uint64_t size;
        XAMG::mpi::probe<T>(nbr, size, comm);
        data.alloc<T>(size);
        data.check(vector::vector::allocated);
        XAMG::mpi::recv<T>(data.get_aligned_ptr<T>(), data.size * NV, nbr, comm);
        data.if_initialized = true;
        data.if_zero = false;
    }
    template <typename T, const uint16_t NV>
    inline void mpi_recv_async(mpi::token &tok, mpi::scope comm) {
        data.check(vector::vector::allocated);
        XAMG::mpi::irecv<T>(data.get_aligned_ptr<T>(), data.size * NV, nbr, tok, comm);
        data.if_initialized = true;
        data.if_zero = false;
    }
    template <typename T, const uint16_t NV>
    inline void mpi_send_async(mpi::token &tok, mpi::scope comm) {
        data.check(vector::vector::initialized);
        XAMG::mpi::isend<T>(data.get_aligned_ptr<T>(), data.size * NV, nbr, tok, comm);
    }
    void set_type(std::type_index type) { data.set_type(type.hash_code()); }
};

struct matvec_comm {

    std::vector<matvec_comm_data> obj;

  private:
    mpi::tokens_ptr _toks;

  public:
    matvec_comm() : _toks() {}
    matvec_comm(const uint32_t nelems) : _toks() {
        for (uint16_t i = 0; i < nelems; ++i) {
            obj.emplace_back(matvec_comm_data(mem::CORE));
        }
        _toks.set_num_tokens(nelems);
    }

    ~matvec_comm() { _toks.free_tokens(); }

    void reserve_tokens(uint32_t n) { _toks.set_num_tokens(n); }

    void alloc_tokens(uint32_t n) { _toks.alloc_tokens(n); }

    void alloc_tokens() { _toks.alloc_tokens(obj.size()); }

    void free_tokens() { _toks.free_tokens(); }

    XAMG::mpi::tokens &get_tokens() {
        assert(_toks.get_num_tokens() == obj.size());
        return (_toks.get_tokens());
    }

    XAMG::mpi::token &get_token(uint32_t i) {
        assert(_toks.get_num_tokens() == obj.size());
        return (_toks.get_token(i));
    }
};

struct comm_p2p {
    std::vector<vector::indx_vector> sender_indx;
    matvec_comm send;
    matvec_comm recv;

    mpi::comm_pool intra_comm_group;
    mpi::comm_pool cross_comm_group;
    mem::sharing sharing_mode;

    comm::sync_one send_syncone;
    comm::sync_all send_syncall;
    comm::sync_one recv_syncone;
    comm::sync_all recv_syncall;
    comm::sync_all send_syncall2;
    comm::sync_one numa_syncone;

    comm::barrier barrier;
    comm::barrier barrier2;

    comm_p2p(const segment::hierarchy layer)
        : intra_comm_group(mpi::intra_layer_comm(layer)),
          cross_comm_group(mpi::cross_layer_comm(layer)),
          sharing_mode(sharing_mode_by_intra_layer(layer)), send_syncone(layer),
          send_syncall(layer), recv_syncone(layer), recv_syncall(layer), send_syncall2(layer),
          numa_syncone(segment::NUMA), barrier(layer), barrier2(layer) {}

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    void create_send_objects(
        const std::vector<matrix::csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks) {
        std::vector<data_exchange_buffer> comm_send_buffer;
        std::vector<data_exchange_buffer> comm_recv_buffer;

        if (cross_comm_group.member_proc) {
            for (uint32_t i = 0; i < csr_chunks.size(); ++i) {
                matrix::const_csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[i]);

                if (chunk_wrapper.proc != (uint32_t)cross_comm_group.proc) {
                    comm_send_buffer.emplace_back(chunk_wrapper.proc);
                    chunk_wrapper.mtx.get_col_ind_vector().template push_to_buffer<I3>(
                        comm_send_buffer.back());
                }
            }
            buffer_alltoall(comm_send_buffer, comm_recv_buffer, cross_comm_group.comm);
        }

        uint32_t nsends = comm_recv_buffer.size();
        mpi::bcast<uint32_t>(&nsends, 1, 0, intra_comm_group.comm);

        for (uint32_t i = 0; i < nsends; ++i) {
            vector::vector temp;
            if (cross_comm_group.member_proc)
                temp.pull_from_buffer<I3>(comm_recv_buffer[i]);
            else
                temp.set_type<I3>();

            uint32_t vec_size = temp.size;
            mpi::bcast<uint32_t>(&vec_size, 1, 0, intra_comm_group.comm);

            sender_indx.emplace_back(sharing_mode);
            sender_indx.back().alloc(vec_size);
            vector::convert_from(temp, sender_indx.back());

            send.obj.emplace_back(matvec_comm_data(sharing_mode));
            if (cross_comm_group.member_proc)
                send.obj.back().nbr = comm_recv_buffer[i].nbr;
            mpi::bcast<uint32_t>(&send.obj.back().nbr, 1, 0, intra_comm_group.comm);
        }
        // for (uint32_t i = 0; i < blocks.size(); ++i)
        //     blocks[i].print();
    }

    template <typename F, typename I1, typename I2, typename I3, typename I4>
    void create_recv_objects(
        const std::vector<matrix::csr_mtx_chunk_pair<F, I1, I2, I3, I4>> &csr_chunks) {
        for (uint32_t i = 0; i < csr_chunks.size(); ++i) {
            matrix::const_csr_mtx_chunk_wrapper<F, I1, I2, I3, I4> chunk_wrapper(csr_chunks[i]);

            if (chunk_wrapper.proc != (uint32_t)cross_comm_group.proc) {
                recv.obj.emplace_back(matvec_comm_data(sharing_mode));
                recv.obj.back().nbr = chunk_wrapper.proc;
            }
        }
    }

    void reset_recv() { recv_syncone.init(); }

    template <typename F, const uint16_t NV>
    void start_recv() {
        recv_syncone.wait(); // all the recv buffers are ready to recv data!
        if (cross_comm_group.member_proc) {
            for (size_t i = 0; i < recv.obj.size(); ++i)
                recv.obj[i].mpi_recv_async<F, NV>(recv.get_token(i), cross_comm_group.comm);
        } else {
            for (size_t i = 0; i < recv.obj.size(); ++i) {
                recv.obj[i].data.if_initialized = true;
                recv.obj[i].data.if_zero = false;
            }
        }
    }

    void finalize_recv() {
        if (cross_comm_group.member_proc) {
            mpi::waitall(recv.get_tokens());
        }
        recv_syncall.init();

        recv_syncall.wait();
    }

    void reset_send() { send_syncall2.init(); }

    template <typename F, const uint16_t NV>
    void start_send(const vector::vector &x) {
        send_syncall2.wait();
        for (size_t i = 0; i < send.obj.size(); ++i)
            // Only numa_master processes involved in case of shared data!
            blas::gather<F, uint32_t, NV>(x, sender_indx[i], send.obj[i].data, true);
        send_syncone.init();

        send_syncone.wait();
        if (cross_comm_group.member_proc) {
            for (size_t i = 0; i < send.obj.size(); ++i)
                send.obj[i].mpi_send_async<F, NV>(send.get_token(i), cross_comm_group.comm);
            // XAMG::out << ALLRANKS << "proc " << id.gl_proc << " sends " << m.node_comm.send.obj[i].data.size << " to " << m.node_comm.send.obj[i].nbr << std::endl;
        }
#ifndef SHM_OPT3
        send_syncall.init();
#endif
    }

    void finalize_send() {
        if (cross_comm_group.member_proc)
            mpi::waitall(send.get_tokens());

#ifndef SHM_OPT3
        send_syncall.wait();
#endif
    }
};

struct comm_global {
    std::vector<vector::vector> buffer;
    mpi::comm_pool comm_group;

    std::vector<int> block_size;
    std::vector<int> block_indx;

    uint16_t nv;

    comm::sync_one numa_syncone;
    comm::sync_one node_syncone;
    comm::sync_all node_syncall;

    comm_global(const segment::hierarchy layer)
        : buffer(0), comm_group(mpi::cross_layer_comm(layer)), block_size(0), block_indx(0), nv(0),
          numa_syncone(segment::NUMA), node_syncone(segment::NODE), node_syncall(segment::NODE) {
        assert(layer == segment::NODE);

        mem::sharing sharing_mode = sharing_mode_by_intra_layer(layer);

        buffer.emplace_back(vector::vector(sharing_mode));
        buffer.emplace_back(vector::vector(sharing_mode));
    }

    vector::vector &local() {
        assert(buffer.size() > 0);
        return buffer[0];
    }

    vector::vector &global() {
        assert(buffer.size() > 1);
        return buffer[1];
    }

    template <typename T, const uint16_t NV>
    void alloc(const std::shared_ptr<part::part> part_) {
        nv = NV;
        uint64_t local_offset = 0;
        if (comm_group.comm == mpi::CROSS_NODE) {
            block_size.resize(part_->node_layer.block_size.size());
            block_indx.resize(part_->node_layer.block_indx.size());
            for (size_t i = 0; i < block_size.size(); ++i)
                block_size[i] = part_->node_layer.block_size[i];
            for (size_t i = 0; i < block_indx.size(); ++i)
                block_indx[i] = part_->node_layer.block_indx[i];

            local_offset = part_->numa_layer.block_indx[id.nd_numa];
        } else if (comm_group.comm == mpi::CROSS_NUMA) {
            assert(0);
        }

        uint64_t local_size = block_size[comm_group.proc];
        uint64_t global_size =
            block_indx[comm_group.nprocs - 1] + block_size[comm_group.nprocs - 1];
        // check if the size of comm vector is not exceeding INT limits:
        assert(global_size < INT_MAX / nv);

        // XAMG::out << XAMG::ALLRANKS << local_size << std::endl;
        buffer[0].alloc<T>(local_size, NV);
        buffer[0].ext_offset = local_offset;
        buffer[1].alloc<T>(global_size, NV);

        // extend by nv to be used in MPI communications
        for (auto &block_elem : block_size)
            block_elem *= nv;
        for (auto &block_elem : block_indx)
            block_elem *= nv;
    }

    template <typename F, const uint16_t NV>
    void fill_buffer(const vector::vector &x) {
        numa_syncone.init();
        numa_syncone.wait();

        // use copy operation to collect NUMA_NODE-shared vector to NODE-shared vector
        blas::copy<F, NV>(x, local());
        node_syncone.init();
    }

    template <typename T, const uint16_t NV>
    void allgatherv() {
        assert(buffer.size() == 2);
        auto &local_vec = local();
        auto &global_vec = global();

        mpi::allgatherv<T>(local_vec.get_aligned_ptr<T>(), nv * local_vec.size,
                           global_vec.get_aligned_ptr<T>(), block_size.data(), block_indx.data(),
                           comm_group.comm);
        global_vec.if_initialized = true;
        global_vec.if_zero = false;
    }

    template <typename T, const uint16_t NV>
    void exchange() {
        node_syncone.wait();
        if (id.numa_master_process()) {
            if (id.node_master_process()) {
                // XAMG::out << ALLRANKS << node_size << " " << node_offset << " " << temp.size << std::endl;
                allgatherv<T, NV>();
            } else {
                // TODO: all flags should be placed in the shared memory
                global().if_initialized = true;
                global().if_zero = false;
            }
        }
        node_syncall.init();
        node_syncall.wait();
    }

    void sync_flags(vector::vector &y) {
        uint8_t flag = y.if_zero;
        mpi::bcast<uint8_t>(&flag, 1, 0, mpi::INTRA_NUMA);
        y.if_initialized = true;
        y.if_zero = flag;
    }
};

} // namespace comm
} // namespace XAMG
