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

#include <mpi.h>
#include "xamg_types.h"
#include "xamg.h"

extern ID id;

#include "comm/mpi_token.h"
#include "comm/detail/mpi_token.inl"

#include "comm/mpi_wrapper.h"
#include "comm/detail/mpi_wrapper.inl"

namespace XAMG {
namespace mpi {

template void isend<uint8_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                             int tag);
template void isend_bigsize<uint8_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                     size_t start_tok, size_t ntoks, scope comm, int tag);
template void send<uint8_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void send_bigsize<uint8_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void irecv<uint8_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                             int tag);
template void irecv_bigsize<uint8_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                     size_t start_tok, size_t ntoks, scope comm, int tag);
template void recv<uint8_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void recv_bigsize<uint8_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void probe<uint8_t>(uint32_t rank, size_t &size, scope comm, int tag);
template void probe_any_source<uint8_t>(uint32_t &rank, size_t &size, scope comm, int tag);
template void allreduce_sum<uint8_t>(uint8_t *in, uint8_t *out, size_t size, scope comm);
template void iallreduce_sum_init<uint8_t>(uint8_t *in, uint8_t *out, size_t size, token &tok,
                                           scope comm);
template void allreduce_min<uint8_t>(uint8_t *in, uint8_t *out, size_t size, scope comm);
template void allreduce_max<uint8_t>(uint8_t *in, uint8_t *out, size_t size, scope comm);
template void iallreduce_max_init<uint8_t>(uint8_t *in, uint8_t *out, size_t size, token &tok,
                                           scope comm);
template void alltoall<uint8_t>(uint8_t *in, size_t size_in, uint8_t *out, size_t size_out,
                                scope comm);
template void gather<uint8_t>(uint8_t *in, size_t size_in, uint8_t *out, size_t size_out,
                              uint32_t root, scope comm);
template void gatherv<uint8_t>(uint8_t *in, size_t size_in, uint8_t *out, int *size_out,
                               int *offset, uint32_t root, scope comm);
template void allgather<uint8_t>(uint8_t *in, size_t size_in, uint8_t *out, size_t size_out,
                                 scope comm);
template void allgatherv<uint8_t>(uint8_t *in, size_t size_in, uint8_t *out, int *size_out,
                                  int *offset, scope comm);
template void scatter<uint8_t>(uint8_t *in, size_t size_in, uint8_t *out, size_t size_out,
                               uint32_t root, scope comm);
template void bcast<uint8_t>(uint8_t *in, size_t size, int root, scope comm);

template void isend<uint16_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                              int tag);
template void isend_bigsize<uint16_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                      size_t start_tok, size_t ntoks, scope comm, int tag);
template void send<uint16_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void send_bigsize<uint16_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void irecv<uint16_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                              int tag);
template void irecv_bigsize<uint16_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                      size_t start_tok, size_t ntoks, scope comm, int tag);
template void recv<uint16_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void recv_bigsize<uint16_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void probe<uint16_t>(uint32_t rank, size_t &size, scope comm, int tag);
template void probe_any_source<uint16_t>(uint32_t &rank, size_t &size, scope comm, int tag);
template void allreduce_sum<uint16_t>(uint16_t *in, uint16_t *out, size_t size, scope comm);
template void iallreduce_sum_init<uint16_t>(uint16_t *in, uint16_t *out, size_t size, token &tok,
                                            scope comm);
template void allreduce_min<uint16_t>(uint16_t *in, uint16_t *out, size_t size, scope comm);
template void allreduce_max<uint16_t>(uint16_t *in, uint16_t *out, size_t size, scope comm);
template void iallreduce_max_init<uint16_t>(uint16_t *in, uint16_t *out, size_t size, token &tok,
                                            scope comm);
template void alltoall<uint16_t>(uint16_t *in, size_t size_in, uint16_t *out, size_t size_out,
                                 scope comm);
template void gather<uint16_t>(uint16_t *in, size_t size_in, uint16_t *out, size_t size_out,
                               uint32_t root, scope comm);
template void gatherv<uint16_t>(uint16_t *in, size_t size_in, uint16_t *out, int *size_out,
                                int *offset, uint32_t root, scope comm);
template void allgather<uint16_t>(uint16_t *in, size_t size_in, uint16_t *out, size_t size_out,
                                  scope comm);
template void allgatherv<uint16_t>(uint16_t *in, size_t size_in, uint16_t *out, int *size_out,
                                   int *offset, scope comm);
template void scatter<uint16_t>(uint16_t *in, size_t size_in, uint16_t *out, size_t size_out,
                                uint32_t root, scope comm);
template void bcast<uint16_t>(uint16_t *in, size_t size, int root, scope comm);

template void isend<uint32_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                              int tag);
template void isend_bigsize<uint32_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                      size_t start_tok, size_t ntoks, scope comm, int tag);
template void send<uint32_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void send_bigsize<uint32_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void irecv<uint32_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                              int tag);
template void irecv_bigsize<uint32_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                      size_t start_tok, size_t ntoks, scope comm, int tag);
template void recv<uint32_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void recv_bigsize<uint32_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void probe<uint32_t>(uint32_t rank, size_t &size, scope comm, int tag);
template void probe_any_source<uint32_t>(uint32_t &rank, size_t &size, scope comm, int tag);
template void allreduce_sum<uint32_t>(uint32_t *in, uint32_t *out, size_t size, scope comm);
template void iallreduce_sum_init<uint32_t>(uint32_t *in, uint32_t *out, size_t size, token &tok,
                                            scope comm);
template void allreduce_min<uint32_t>(uint32_t *in, uint32_t *out, size_t size, scope comm);
template void allreduce_max<uint32_t>(uint32_t *in, uint32_t *out, size_t size, scope comm);
template void iallreduce_max_init<uint32_t>(uint32_t *in, uint32_t *out, size_t size, token &tok,
                                            scope comm);
template void alltoall<uint32_t>(uint32_t *in, size_t size_in, uint32_t *out, size_t size_out,
                                 scope comm);
template void gather<uint32_t>(uint32_t *in, size_t size_in, uint32_t *out, size_t size_out,
                               uint32_t root, scope comm);
template void gatherv<uint32_t>(uint32_t *in, size_t size_in, uint32_t *out, int *size_out,
                                int *offset, uint32_t root, scope comm);
template void allgather<uint32_t>(uint32_t *in, size_t size_in, uint32_t *out, size_t size_out,
                                  scope comm);
template void allgatherv<uint32_t>(uint32_t *in, size_t size_in, uint32_t *out, int *size_out,
                                   int *offset, scope comm);
template void scatter<uint32_t>(uint32_t *in, size_t size_in, uint32_t *out, size_t size_out,
                                uint32_t root, scope comm);
template void bcast<uint32_t>(uint32_t *in, size_t size, int root, scope comm);

template void isend<uint64_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                              int tag);
template void isend_bigsize<uint64_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                      size_t start_tok, size_t ntoks, scope comm, int tag);
template void send<uint64_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void send_bigsize<uint64_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void irecv<uint64_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                              int tag);
template void irecv_bigsize<uint64_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                      size_t start_tok, size_t ntoks, scope comm, int tag);
template void recv<uint64_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void recv_bigsize<uint64_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void probe<uint64_t>(uint32_t rank, size_t &size, scope comm, int tag);
template void probe_any_source<uint64_t>(uint32_t &rank, size_t &size, scope comm, int tag);
template void allreduce_sum<uint64_t>(uint64_t *in, uint64_t *out, size_t size, scope comm);
template void iallreduce_sum_init<uint64_t>(uint64_t *in, uint64_t *out, size_t size, token &tok,
                                            scope comm);
template void allreduce_min<uint64_t>(uint64_t *in, uint64_t *out, size_t size, scope comm);
template void allreduce_max<uint64_t>(uint64_t *in, uint64_t *out, size_t size, scope comm);
template void iallreduce_max_init<uint64_t>(uint64_t *in, uint64_t *out, size_t size, token &tok,
                                            scope comm);
template void alltoall<uint64_t>(uint64_t *in, size_t size_in, uint64_t *out, size_t size_out,
                                 scope comm);
template void gather<uint64_t>(uint64_t *in, size_t size_in, uint64_t *out, size_t size_out,
                               uint32_t root, scope comm);
template void gatherv<uint64_t>(uint64_t *in, size_t size_in, uint64_t *out, int *size_out,
                                int *offset, uint32_t root, scope comm);
template void allgather<uint64_t>(uint64_t *in, size_t size_in, uint64_t *out, size_t size_out,
                                  scope comm);
template void allgatherv<uint64_t>(uint64_t *in, size_t size_in, uint64_t *out, int *size_out,
                                   int *offset, scope comm);
template void scatter<uint64_t>(uint64_t *in, size_t size_in, uint64_t *out, size_t size_out,
                                uint32_t root, scope comm);
template void bcast<uint64_t>(uint64_t *in, size_t size, int root, scope comm);

template void isend<float32_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                               int tag);
template void isend_bigsize<float32_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                       size_t start_tok, size_t ntoks, scope comm, int tag);
template void send<float32_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void send_bigsize<float32_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void irecv<float32_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                               int tag);
template void irecv_bigsize<float32_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                       size_t start_tok, size_t ntoks, scope comm, int tag);
template void recv<float32_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void recv_bigsize<float32_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void probe<float32_t>(uint32_t rank, size_t &size, scope comm, int tag);
template void probe_any_source<float32_t>(uint32_t &rank, size_t &size, scope comm, int tag);
template void allreduce_sum<float32_t>(float32_t *in, float32_t *out, size_t size, scope comm);
template void iallreduce_sum_init<float32_t>(float32_t *in, float32_t *out, size_t size, token &tok,
                                             scope comm);
template void allreduce_min<float32_t>(float32_t *in, float32_t *out, size_t size, scope comm);
template void allreduce_max<float32_t>(float32_t *in, float32_t *out, size_t size, scope comm);
template void iallreduce_max_init<float32_t>(float32_t *in, float32_t *out, size_t size, token &tok,
                                             scope comm);
template void alltoall<float32_t>(float32_t *in, size_t size_in, float32_t *out, size_t size_out,
                                  scope comm);
template void gather<float32_t>(float32_t *in, size_t size_in, float32_t *out, size_t size_out,
                                uint32_t root, scope comm);
template void gatherv<float32_t>(float32_t *in, size_t size_in, float32_t *out, int *size_out,
                                 int *offset, uint32_t root, scope comm);
template void allgather<float32_t>(float32_t *in, size_t size_in, float32_t *out, size_t size_out,
                                   scope comm);
template void allgatherv<float32_t>(float32_t *in, size_t size_in, float32_t *out, int *size_out,
                                    int *offset, scope comm);
template void scatter<float32_t>(float32_t *in, size_t size_in, float32_t *out, size_t size_out,
                                 uint32_t root, scope comm);
template void bcast<float32_t>(float32_t *in, size_t size, int root, scope comm);

template void isend<float64_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                               int tag);
template void isend_bigsize<float64_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                       size_t start_tok, size_t ntoks, scope comm, int tag);
template void send<float64_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void send_bigsize<float64_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void irecv<float64_t>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm,
                               int tag);
template void irecv_bigsize<float64_t>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                       size_t start_tok, size_t ntoks, scope comm, int tag);
template void recv<float64_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void recv_bigsize<float64_t>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void probe<float64_t>(uint32_t rank, size_t &size, scope comm, int tag);
template void probe_any_source<float64_t>(uint32_t &rank, size_t &size, scope comm, int tag);
template void allreduce_sum<float64_t>(float64_t *in, float64_t *out, size_t size, scope comm);
template void iallreduce_sum_init<float64_t>(float64_t *in, float64_t *out, size_t size, token &tok,
                                             scope comm);
template void allreduce_min<float64_t>(float64_t *in, float64_t *out, size_t size, scope comm);
template void allreduce_max<float64_t>(float64_t *in, float64_t *out, size_t size, scope comm);
template void iallreduce_max_init<float64_t>(float64_t *in, float64_t *out, size_t size, token &tok,
                                             scope comm);
template void alltoall<float64_t>(float64_t *in, size_t size_in, float64_t *out, size_t size_out,
                                  scope comm);
template void gather<float64_t>(float64_t *in, size_t size_in, float64_t *out, size_t size_out,
                                uint32_t root, scope comm);
template void gatherv<float64_t>(float64_t *in, size_t size_in, float64_t *out, int *size_out,
                                 int *offset, uint32_t root, scope comm);
template void allgather<float64_t>(float64_t *in, size_t size_in, float64_t *out, size_t size_out,
                                   scope comm);
template void allgatherv<float64_t>(float64_t *in, size_t size_in, float64_t *out, int *size_out,
                                    int *offset, scope comm);
template void scatter<float64_t>(float64_t *in, size_t size_in, float64_t *out, size_t size_out,
                                 uint32_t root, scope comm);
template void bcast<float64_t>(float64_t *in, size_t size, int root, scope comm);

template void isend<int>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm, int tag);
template void isend_bigsize<int>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                 size_t start_tok, size_t ntoks, scope comm, int tag);
template void send<int>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void send_bigsize<int>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void irecv<int>(void *ptr, size_t size, uint32_t rank, token &tok, scope comm, int tag);
template void irecv_bigsize<int>(void *ptr, size_t size, uint32_t rank, tokens &tok,
                                 size_t start_tok, size_t ntoks, scope comm, int tag);
template void recv<int>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void recv_bigsize<int>(void *ptr, size_t size, uint32_t rank, scope comm, int tag);
template void probe<int>(uint32_t rank, size_t &size, scope comm, int tag);
template void probe_any_source<int>(uint32_t &rank, size_t &size, scope comm, int tag);
template void allreduce_sum<int>(int *in, int *out, size_t size, scope comm);
template void iallreduce_sum_init<int>(int *in, int *out, size_t size, token &tok, scope comm);
template void allreduce_min<int>(int *in, int *out, size_t size, scope comm);
template void allreduce_max<int>(int *in, int *out, size_t size, scope comm);
template void iallreduce_max_init<int>(int *in, int *out, size_t size, token &tok, scope comm);
template void alltoall<int>(int *in, size_t size_in, int *out, size_t size_out, scope comm);
template void gather<int>(int *in, size_t size_in, int *out, size_t size_out, uint32_t root,
                          scope comm);
template void gatherv<int>(int *in, size_t size_in, int *out, int *size_out, int *offset,
                           uint32_t root, scope comm);
template void allgather<int>(int *in, size_t size_in, int *out, size_t size_out, scope comm);
template void allgatherv<int>(int *in, size_t size_in, int *out, int *size_out, int *offset,
                              scope comm);
template void scatter<int>(int *in, size_t size_in, int *out, size_t size_out, uint32_t root,
                           scope comm);
template void bcast<int>(int *in, size_t size, int root, scope comm);

} // namespace mpi
} // namespace XAMG
