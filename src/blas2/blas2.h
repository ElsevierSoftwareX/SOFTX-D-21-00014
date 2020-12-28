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

#include "xamg_types.h"

#include "blas/blas.h"

#include "comm/matvec_comm.h"
#include "primitives/matrix/matrix.h"

#include "matvec_backend.h"

#include "comm/shm_sync.h"

namespace XAMG {
namespace blas2 {

template <const uint16_t NV>
void generate_blas2_drivers(matrix::matrix &m);

template <typename F, const uint16_t NV>
void Axpy_block(matrix::matrix &m, const vector::vector &x, vector::vector &y, uint16_t nv,
                XAMG::mpi::token &tok = XAMG::mpi::null_token) {
    //    assert (tok == XAMG::mpi::null_token);
    auto &numa_layer = m.data_layer.find(segment::NUMA)->second;
    auto &node_layer = m.data_layer.find(segment::NODE)->second;
    assert(numa_layer.p2p_comm.recv.obj.size() == numa_layer.offd.size());
    assert(node_layer.p2p_comm.recv.obj.size() == node_layer.offd.size());

    numa_layer.p2p_comm.barrier.init();

    node_layer.p2p_comm.start_recv<F, NV>();
    numa_layer.p2p_comm.start_recv<F, NV>();

    // this check is needed to ensure all intra-numa procs have finished x-vector modifications outside the SpMV
    numa_layer.p2p_comm.barrier.wait();
    node_layer.p2p_comm.start_send<F, NV>(x);
    numa_layer.p2p_comm.start_send<F, NV>(x);

    ////

    numa_layer.diag.data->Axpy(numa_layer.diag.blas2_driver, x, y, nv);
    numa_layer.p2p_comm.barrier2.init();

    numa_layer.p2p_comm.finalize_recv<F>();
    for (uint32_t i = 0; i < numa_layer.offd.size(); ++i) {
        numa_layer.offd[i].data->Axpy(numa_layer.offd[i].blas2_driver,
                                      numa_layer.p2p_comm.recv.obj[i].data, y, nv);
        //    numa_layer.p2p_comm.recv_syncone.init();
    }
    numa_layer.p2p_comm.reset_recv();

    node_layer.p2p_comm.finalize_recv<F>();
    for (uint32_t i = 0; i < node_layer.offd.size(); ++i)
        node_layer.offd[i].data->Axpy(node_layer.offd[i].blas2_driver,
                                      node_layer.p2p_comm.recv.obj[i].data, y, nv);
    //    node_layer.p2p_comm.recv_syncone.init();
    node_layer.p2p_comm.reset_recv();

    numa_layer.p2p_comm.finalize_send();
    numa_layer.p2p_comm.reset_send();
    node_layer.p2p_comm.finalize_send();
    node_layer.p2p_comm.reset_send();

    // this check is needed to ensure all intra-numa procs have finished read ops of x-vector inside the SpMV
    numa_layer.p2p_comm.barrier2.wait();

    //    mpi::barrier(mpi::INTRA_NUMA);
    //    XAMG::out.norm<F, NV>(y, "y_total");
}

template <typename F, const uint16_t NV>
void Ax_y_block(matrix::matrix &m, const vector::vector &x, vector::vector &y, uint16_t nv,
                XAMG::mpi::token &tok = XAMG::mpi::null_token) {
    //    assert (tok == XAMG::mpi::null_token);
    auto &numa_layer = m.data_layer.find(segment::NUMA)->second;
    auto &node_layer = m.data_layer.find(segment::NODE)->second;
    assert(numa_layer.p2p_comm.recv.obj.size() == numa_layer.offd.size());
    assert(node_layer.p2p_comm.recv.obj.size() == node_layer.offd.size());

    numa_layer.p2p_comm.barrier.init();

    //    XAMG::out.norm<F, NV>(x, "X init");
    node_layer.p2p_comm.start_recv<F, NV>();
    numa_layer.p2p_comm.start_recv<F, NV>();

    numa_layer.p2p_comm.barrier.wait();
    node_layer.p2p_comm.start_send<F, NV>(x);
    numa_layer.p2p_comm.start_send<F, NV>(x);

    ////

    numa_layer.diag.data->Ax_y(numa_layer.diag.blas2_driver, x, y, nv);
    numa_layer.p2p_comm.barrier2.init();

    numa_layer.p2p_comm.finalize_recv<F>();
    for (uint32_t i = 0; i < numa_layer.offd.size(); ++i) {
        numa_layer.offd[i].data->Axpy(numa_layer.offd[i].blas2_driver,
                                      numa_layer.p2p_comm.recv.obj[i].data, y, nv);
    }
    numa_layer.p2p_comm.reset_recv();

    node_layer.p2p_comm.finalize_recv<F>();
    for (uint32_t i = 0; i < node_layer.offd.size(); ++i) {
        node_layer.offd[i].data->Axpy(node_layer.offd[i].blas2_driver,
                                      node_layer.p2p_comm.recv.obj[i].data, y, nv);
    }
    node_layer.p2p_comm.reset_recv();

    numa_layer.p2p_comm.finalize_send();
    numa_layer.p2p_comm.reset_send();
    node_layer.p2p_comm.finalize_send();
    node_layer.p2p_comm.reset_send();

    numa_layer.p2p_comm.barrier2.wait();

    //    mpi::barrier(mpi::INTRA_NUMA);
    //    XAMG::out.norm<F, NV>(y, "y_total");
}

//----------------------

template <typename F, const uint16_t NV>
void Axpy_slice(matrix::matrix &m, const vector::vector &x, vector::vector &y, uint16_t nv,
                XAMG::mpi::token &tok = XAMG::mpi::null_token) {
    assert(0 && "Unsupported...");
}

template <typename F, const uint16_t NV>
void Ax_y_slice(matrix::matrix &m, const vector::vector &x, vector::vector &y, uint16_t nv,
                XAMG::mpi::token &tok = XAMG::mpi::null_token) {
    assert(m.data_layer.find(segment::NUMA) != m.data_layer.end());
    assert(x.sharing_mode == mem::NUMA_NODE);
    auto &numa_layer = m.data_layer.find(segment::NUMA)->second;
    assert(numa_layer.offd.size() == 0);
    auto &global_comm = m.global_comm;

    /*
        numa_layer.global_comm.numa_syncone.init();
        numa_layer.global_comm.numa_syncone.wait();
    //    use copy operation to collect NUMA_NODE-shared vector to NODE-shared vector
        blas::copy<F, NV>(x, numa_layer.global_comm.local());
        numa_layer.global_comm.node_syncone.init();
    */
    global_comm.fill_buffer<F, NV>(x);
    global_comm.exchange<F, NV>();

    if (id.numa_master_process())
        numa_layer.diag.data->Ax_y(numa_layer.diag.blas2_driver, global_comm.global(), y, nv);

    global_comm.sync_flags(y);
}

//----------------------

template <typename F, const uint16_t NV>
void SGS(matrix::matrix &m, const vector::vector &b, vector::vector &x, vector::vector &t,
         const vector::vector &relax_factor, uint16_t nv,
         XAMG::mpi::token &tok = XAMG::mpi::null_token) {
    monitor.start("hsgs_loop");
    monitor.start("hsgs_serv");
    assert(m.segmentation == matrix::SEGMENT_BY_BLOCKS);
    if (!m.if_drivers_allocated)
        generate_blas2_drivers<NV>(m);
    if (!m.if_buffers_allocated)
        m.alloc_comm_buffers<F, NV>();

    //    assert (tok == XAMG::mpi::null_token);
    auto &core_layer = m.data_layer.find(segment::CORE)->second;
    auto &numa_layer = m.data_layer.find(segment::NUMA)->second;
    auto &node_layer = m.data_layer.find(segment::NODE)->second;
    assert(node_layer.p2p_comm.recv.obj.size() == node_layer.offd.size());
    assert(numa_layer.p2p_comm.recv.obj.size() == numa_layer.offd.size());
    monitor.stop("hsgs_serv");

    //    numa_layer.p2p_comm.numa_syncone.init();

    monitor.start("hsgs_irecv");
    numa_layer.p2p_comm.barrier.init();

    //    XAMG::out << XAMG::DBG;
    //    XAMG::out.norm<F, NV>(x, "HSGS: X init");

    node_layer.p2p_comm.start_recv<F, NV>();
    numa_layer.p2p_comm.start_recv<F, NV>();
    core_layer.p2p_comm.start_recv<F, NV>();
    monitor.stop("hsgs_irecv");

    //    numa_layer.p2p_comm.numa_syncone.wait();
    monitor.start("hsgs_isend");
    numa_layer.p2p_comm.barrier.wait();
    node_layer.p2p_comm.start_send<F, NV>(x);
    numa_layer.p2p_comm.start_send<F, NV>(x);
    core_layer.p2p_comm.start_send<F, NV>(x);
    monitor.stop("hsgs_isend");

    monitor.start("hsgs_core_offd");
    blas::set_const<F, NV>(t, 0.0, true);

    core_layer.p2p_comm.finalize_recv<F>();
    for (uint32_t i = 0; i < core_layer.offd.size(); ++i) {
        core_layer.offd[i].data->Axpy(core_layer.offd[i].blas2_driver,
                                      core_layer.p2p_comm.recv.obj[i].data, t, nv);
    }
    core_layer.p2p_comm.reset_recv();
    monitor.stop("hsgs_core_offd");

    monitor.start("hsgs_numa_offd");
    numa_layer.p2p_comm.finalize_recv<F>();
    for (uint32_t i = 0; i < numa_layer.offd.size(); ++i) {
        numa_layer.offd[i].data->Axpy(numa_layer.offd[i].blas2_driver,
                                      numa_layer.p2p_comm.recv.obj[i].data, t, nv);
    }
    numa_layer.p2p_comm.reset_recv();
    monitor.stop("hsgs_numa_offd");

    monitor.start("hsgs_node_offd");
    node_layer.p2p_comm.finalize_recv<F>();
    for (uint32_t i = 0; i < node_layer.offd.size(); ++i) {
        node_layer.offd[i].data->Axpy(node_layer.offd[i].blas2_driver,
                                      node_layer.p2p_comm.recv.obj[i].data, t, nv);
    }
    node_layer.p2p_comm.reset_recv();
    monitor.stop("hsgs_node_offd");

    //////////

    monitor.start("hsgs_diag");
    blas::axpby<F, NV>(1.0, b, -1.0, t);

    if (x.if_zero)
        blas::set_const<F, NV>(x, 0.0, true);
    core_layer.diag.data->SGS(core_layer.diag.blas2_driver, m.inv_diag(), t, x, relax_factor, nv);
    monitor.stop("hsgs_diag");

    /////////

    //    XAMG::out << XAMG::DBG;
    //    XAMG::out.norm<F, NV>(x, "HSGS: X end");

    monitor.start("hsgs_fin_comm");
    core_layer.p2p_comm.finalize_send();
    core_layer.p2p_comm.reset_send();
    numa_layer.p2p_comm.finalize_send();
    numa_layer.p2p_comm.reset_send();
    node_layer.p2p_comm.finalize_send();
    node_layer.p2p_comm.reset_send();
    monitor.stop("hsgs_fin_comm");

    monitor.stop("hsgs_loop");
}

template <typename F, const uint16_t NV>
void Axpy(matrix::matrix &m, const vector::vector &x, vector::vector &y, uint16_t nv,
          XAMG::mpi::token &tok = XAMG::mpi::null_token) {
    assert(x.type_hash == y.type_hash);
    x.sync_zero_flag();
    if (x.if_zero) {
        return;
    }
    if (!m.if_drivers_allocated)
        generate_blas2_drivers<NV>(m);
    if (!m.if_buffers_allocated)
        m.alloc_comm_buffers<F, NV>();

    if (m.segmentation == matrix::SEGMENT_BY_BLOCKS)
        Axpy_block<F, NV>(m, x, y, nv, tok);
    else if (m.segmentation == matrix::SEGMENT_BY_SLICES)
        Axpy_slice<F, NV>(m, x, y, nv, tok);
    else {
        assert(0 && "Incorrect data segmentation type");
    }
}

template <typename F, const uint16_t NV>
void Ax_y(matrix::matrix &m, const vector::vector &x, vector::vector &y, uint16_t nv,
          XAMG::mpi::token &tok = XAMG::mpi::null_token) {
    assert(x.type_hash == y.type_hash);
    x.sync_zero_flag();
    if (x.if_zero) {
        y.if_zero = true;
        return;
    }
    if (!m.if_drivers_allocated)
        generate_blas2_drivers<NV>(m);
    if (!m.if_buffers_allocated)
        m.alloc_comm_buffers<F, NV>();

    if (m.segmentation == matrix::SEGMENT_BY_BLOCKS) {
        Ax_y_block<F, NV>(m, x, y, nv, tok);
    } else if (m.segmentation == matrix::SEGMENT_BY_SLICES) {
        Ax_y_slice<F, NV>(m, x, y, nv, tok);
    } else {
        assert(0 && "Incorrect data segmentation type");
    }
}

} // namespace blas2
} // namespace XAMG

#include "detail/gen/blas2_driver.inl"
