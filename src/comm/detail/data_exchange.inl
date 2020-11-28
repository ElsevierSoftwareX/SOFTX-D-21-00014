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
namespace comm {

static inline void buffer_alltoall(std::vector<data_exchange_buffer> &send_buffer,
                                   std::vector<data_exchange_buffer> &recv_buffer,
                                   mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    std::vector<uint64_t> send_buffer_size(comm_group.nprocs, 0);
    std::vector<uint64_t> recv_buffer_size(comm_group.nprocs, 0);

    for (auto &buf : send_buffer)
        send_buffer_size[buf.nbr] = buf.global_buf.size();

    mpi::alltoall<uint64_t>(send_buffer_size.data(), 1, recv_buffer_size.data(), 1,
                            comm_group.comm);

    std::vector<int> tokens_per_comm;
    uint32_t recv_buf = 0, ntoks = 0;
    for (uint32_t np = 0; np < (uint32_t)comm_group.nprocs; ++np) {
        if (recv_buffer_size[np]) {
            tokens_per_comm.push_back(mpi::num_parts_for_message(recv_buffer_size[np]));
            ntoks += tokens_per_comm.back();
            recv_buf++;
        }
    }
    recv_buffer.resize(recv_buf, data_exchange_buffer(mem::LOCAL));
    mpi::tokens recv_toks(ntoks);

    recv_buf = 0;
    size_t tok = 0;
    for (uint32_t np = 0; np < (uint32_t)comm_group.nprocs; ++np) {
        if (recv_buffer_size[np]) {
            auto &b = recv_buffer[recv_buf];
            auto size = recv_buffer_size[np];
            auto ntokens = tokens_per_comm[recv_buf];
            b.global_buf.resize(size);
            auto data = b.global_buf.data();
            if (ntokens == 1) { // FIXME go with bigsize version all the time?
                mpi::irecv<uint8_t>(data, size, np, recv_toks[tok], comm_group.comm);
            } else {
                mpi::irecv_bigsize<uint8_t>(data, size, np, recv_toks, tok, ntokens,
                                            comm_group.comm);
            }
            b.nbr = np;
            tok += ntokens;
            recv_buf++;
        }
    }

    for (auto &buf : send_buffer) {
        mpi::send_bigsize<uint8_t>(buf.global_buf.data(), buf.global_buf.size(), buf.nbr,
                                   comm_group.comm);
    }

    mpi::waitall(recv_toks);
}

static inline void buffer_gather(data_exchange_buffer &send_buffer,
                                 std::vector<data_exchange_buffer> &recv_buffer,
                                 const uint32_t root, mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    assert(send_buffer.nbr == root); // probably this check should be removed
    uint64_t send_buffer_size = send_buffer.global_buf.size();
    std::vector<uint64_t> recv_buffer_size;

    if (comm_group.proc == (int)root) {
        recv_buffer_size.resize(comm_group.nprocs, 0);
        recv_buffer.resize(comm_group.nprocs, data_exchange_buffer(mem::LOCAL));
    }
    mpi::gather<uint64_t>(&send_buffer_size, 1, recv_buffer_size.data(), 1, send_buffer.nbr,
                          comm_group.comm);
    std::vector<int> tokens_per_comm;
    uint32_t ntoks = 0;
    if (comm_group.proc == (int)root) {
        for (uint32_t np = 0; np < (uint32_t)comm_group.nprocs; ++np) {
            tokens_per_comm.push_back(mpi::num_parts_for_message(recv_buffer_size[np]));
            ntoks += tokens_per_comm.back();
        }
    }
    mpi::tokens recv_toks(ntoks);

    /////////

    //    mpi::allgatherv<uint8_t>(send_buffer.buf.data(), send_buffer_size, recv_buffer.data(), 1);

    size_t tok = 0;
    if (comm_group.proc == (int)root) {
        for (uint16_t nb = 0; nb < comm_group.nprocs; ++nb) {
            auto &b = recv_buffer[nb];
            auto size = recv_buffer_size[nb];
            auto ntokens = tokens_per_comm[nb];
            b.global_buf.resize(size);
            auto data = b.global_buf.data();
            if (ntokens == 1) { // FIXME go with bigsize version all the time?
                mpi::irecv<uint8_t>(data, size, nb, recv_toks[tok], comm_group.comm);
            } else {
                mpi::irecv_bigsize<uint8_t>(data, size, nb, recv_toks, tok, ntokens,
                                            comm_group.comm);
            }
            recv_buffer[nb].nbr = nb;
            tok += ntokens;
        }
    }

    auto size = send_buffer.global_buf.size();
    auto data = send_buffer.global_buf.data();
    mpi::send_bigsize<uint8_t>(data, size, send_buffer.nbr, comm_group.comm);
    if (comm_group.proc == (int)root) {
        mpi::waitall(recv_toks);
    }
}

static inline void buffer_scatter(std::vector<data_exchange_buffer> &send_buffer,
                                  data_exchange_buffer &recv_buffer, const uint32_t root,
                                  mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    std::vector<uint64_t> send_buffer_size;
    for (auto &buf : send_buffer)
        send_buffer_size.emplace_back(buf.global_buf.size());

    uint64_t recv_buffer_size;
    mpi::scatter<uint64_t>(send_buffer_size.data(), 1, &recv_buffer_size, 1, root, comm_group.comm);

    /////////

    auto size = recv_buffer_size;
    recv_buffer.global_buf.resize(recv_buffer_size);
    auto data = recv_buffer.global_buf.data();
    int ntoks = mpi::num_parts_for_message(size);
    mpi::tokens recv_toks(ntoks);
    if (ntoks == 1) { // FIXME go with bigsize version all the time?
        mpi::irecv<uint8_t>(data, size, 0, recv_toks[0], comm_group.comm);
    } else {
        mpi::irecv_bigsize<uint8_t>(data, size, 0, recv_toks, 0, ntoks, comm_group.comm);
    }
    recv_buffer.nbr = 0;

    uint32_t recv_buf = 0;
    if (comm_group.proc == (int)root) {
        for (uint32_t np = 0; np < (uint32_t)comm_group.nprocs; ++np) {
            auto size = send_buffer[np].global_buf.size();
            auto data = send_buffer[np].global_buf.data();
            mpi::send_bigsize<uint8_t>(data, size, np, comm_group.comm);
        }
    }

    mpi::waitall(recv_toks);
}

static inline void buffer_bcast(data_exchange_buffer &buffer, const uint32_t root,
                                mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    uint64_t buffer_size = buffer.global_buf.size();
    mpi::bcast<uint64_t>(&buffer_size, 1, root, comm_group.comm);

    if (comm_group.proc != (int)root)
        buffer.global_buf.resize(buffer_size);

    mpi::bcast<uint8_t>(buffer.global_buf.data(), buffer.global_buf.size(), root, comm_group.comm);
    mpi::bcast<uint32_t>(&buffer.nbr, 1, root, comm_group.comm);
}

static inline void buffer_bcast(std::vector<data_exchange_buffer> &buffer, const uint32_t root,
                                mpi::scope comm) {
    mpi::comm_pool comm_group(comm);

    if (comm_group.proc != (int)root)
        buffer.resize(comm_group.nprocs);

    for (auto &buf : buffer)
        buffer_bcast(buf, root, comm_group.comm);
}

static inline void buffer_allgather(data_exchange_buffer &send_buffer,
                                    std::vector<data_exchange_buffer> &recv_buffer,
                                    mpi::scope comm) {
    buffer_gather(send_buffer, recv_buffer, 0, comm);
    buffer_bcast(recv_buffer, 0, comm);
}

} // namespace comm
} // namespace XAMG
