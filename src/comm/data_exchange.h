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

#include "comm/mpi_wrapper.h"

namespace XAMG {
namespace comm {

struct data_exchange_buffer {
    std::vector<uint8_t> local_buf;
    std::vector<uint8_t> global_buf;
    uint64_t offset;
    uint32_t nbr;

    data_exchange_buffer() : offset(0), nbr(0) {}
    data_exchange_buffer(uint32_t nbr_) : offset(0), nbr(nbr_) {}

    template <typename T>
    void push_scalar(const T &elem) {
        uint64_t buf_size = global_buf.size();
        uint64_t elem_size = sizeof(T);
        global_buf.resize(offset + elem_size);

        memcpy(global_buf.data() + offset, &elem, elem_size);
        offset += elem_size;
    }

    template <typename T>
    void pull_scalar(T &elem) {
        uint64_t buf_size = global_buf.size();
        uint64_t elem_size = sizeof(T);
        assert(buf_size >= offset + elem_size);

        memcpy(&elem, global_buf.data() + offset, elem_size);
        offset += elem_size;
    }

    template <typename T>
    void push_array(const T *elem, const uint64_t size) {
        uint64_t buf_size = global_buf.size();
        uint64_t elem_size = sizeof(T) * size;
        global_buf.resize(offset + elem_size);

        memcpy(global_buf.data() + offset, elem, elem_size);
        offset += elem_size;
    }

    template <typename T>
    void pull_array(T *elem, const uint64_t size) {
        uint64_t buf_size = global_buf.size();
        uint64_t elem_size = size * sizeof(T);
        assert(buf_size >= offset + elem_size);

        memcpy(elem, global_buf.data() + offset, elem_size);
        offset += elem_size;
    }
};

static inline void buffer_alltoall(std::vector<data_exchange_buffer> &send_buffer,
                                   std::vector<data_exchange_buffer> &recv_buffer,
                                   mpi::scope comm = mpi::GLOBAL);

static inline void buffer_gather(data_exchange_buffer &send_buffer,
                                 std::vector<data_exchange_buffer> &recv_buffer,
                                 const uint32_t root, mpi::scope comm = mpi::GLOBAL);

static inline void buffer_scatter(std::vector<data_exchange_buffer> &send_buffer,
                                  data_exchange_buffer &recv_buffer, const uint32_t root,
                                  mpi::scope comm = mpi::GLOBAL);

static inline void buffer_bcast(data_exchange_buffer &buffer, const uint32_t root,
                                mpi::scope comm = mpi::GLOBAL);

static inline void buffer_bcast(std::vector<data_exchange_buffer> &buffer, const uint32_t root,
                                mpi::scope comm = mpi::GLOBAL);

static inline void buffer_allgather(data_exchange_buffer &send_buffer,
                                    std::vector<data_exchange_buffer> &recv_buffer,
                                    mpi::scope comm = mpi::GLOBAL);
} // namespace comm
} // namespace XAMG

#include "detail/data_exchange.inl"
