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

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#if (defined(__GNUC__) && __GNUC__ < 5) || (defined(__GNUG__) && __GNUG__ < 5)
#error Minimal supported version of GNU C++ compiler is 5.0.0: __GNUC__  __GNUG__
#endif
#endif

#include "xamg_headers.h"

///////////////////

#define sigNaN (std::numeric_limits<float64_t>::signaling_NaN())

///////////////////

// typedef float16 float16;
typedef float float32_t;
typedef double float64_t;

///////////////////
//  bit masks

const uint16_t I8_TYPE = 0;
const uint16_t I16_TYPE = 1;
const uint16_t I32_TYPE = 2;
const uint16_t I64_TYPE = 3;

// const uint16_t F16_TYPE = 0;
// const uint16_t F32_TYPE = 1;
// const uint16_t F64_TYPE = 2;
// const uint16_t F128_TYPE = 3;

const uint16_t F16_TYPE = 4;
const uint16_t F32_TYPE = 5;
const uint16_t F64_TYPE = 6;
const uint16_t F128_TYPE = 7;

const uint16_t NBITS = 3;

const uint16_t I1_OFFSET = 3;
const uint16_t I2_OFFSET = 5;
const uint16_t I3_OFFSET = 7;
const uint16_t I4_OFFSET = 9;

const uint16_t F_MASK = 7;
const uint16_t I1_MASK = (NBITS << I1_OFFSET);
const uint16_t I2_MASK = (NBITS << I2_OFFSET);
const uint16_t I3_MASK = (NBITS << I3_OFFSET);
const uint16_t I4_MASK = (NBITS << I4_OFFSET);

#define XAMG_ALIGN_SIZE 64
#define XAMG_STACK_ALIGNMENT_PREFIX alignas(XAMG_ALIGN_SIZE)

///////////////////
namespace XAMG {

namespace segment {
enum hierarchy { NODE, NUMA, CORE };
}

enum multigrid_cycle { V_cycle = 0, F_cycle = 1, W_cycle = 2 };

enum storage_format { dense, csr, ell };

namespace mem {
enum allocation { LOCAL, SHARED, DISTRIBUTED };
enum sharing { NODE, NUMA, NUMA_NODE, CORE };

} // namespace mem

static inline mem::sharing sharing_mode_by_intra_layer(const segment::hierarchy layer) {
    mem::sharing sharing_mode = mem::CORE;

    if (layer == segment::NODE) {
        sharing_mode = mem::NODE;
    } else if (layer == segment::NUMA) {
        sharing_mode = mem::NUMA;
    } else if (layer == segment::CORE) {
        sharing_mode = mem::CORE;
    } else {
        assert(0);
    }

    return sharing_mode;
}

static inline segment::hierarchy intra_layer_by_sharing_mode(const mem::sharing sharing_mode) {
    segment::hierarchy layer = segment::CORE;

    if (sharing_mode == mem::NODE) {
        layer = segment::NODE;
    } else if (sharing_mode == mem::NUMA) {
        layer = segment::NUMA;
    } else if (sharing_mode == mem::CORE) {
        layer = segment::CORE;
    } else {
        assert(0);
    }

    return layer;
}

} // namespace XAMG
///////////////////

struct ID {
    int gl_proc, gl_nprocs;
    int gl_node, gl_nnodes;
    int nd_numa, nd_nnumas;
    int nd_core, nd_ncores;
    int nm_core, nm_ncores;

    void *mpi_details;

    uint32_t exec_id;
    int numa_sem_id, node_sem_id;
    int numa_shm_id, node_shm_id;

    //    Templates for memory allocator; not used yet
    void *shm_pool;
    void *sem_pool;

    ID()
        : gl_proc(-1), gl_nprocs(0), gl_node(-1), gl_nnodes(0), nd_numa(-1), nd_nnumas(0),
          nd_core(-1), nd_ncores(0), nm_core(-1), nm_ncores(0), mpi_details(nullptr), exec_id(0),
          numa_sem_id(0), node_sem_id(0), numa_shm_id(0), node_shm_id(0), shm_pool(nullptr),
          sem_pool(nullptr) {}

    void *get_comm();
    void *get_comm(const XAMG::segment::hierarchy layer);

    bool numa_master_process() {
        if (!nm_core)
            return true;
        else
            return false;
    }

    bool numa_slave_process() { return (!numa_master_process()); }

    bool node_master_process() {
        if (!nd_core)
            return true;
        else
            return false;
    }

    bool node_slave_process() { return (!node_master_process()); }

    bool master_process() {
        if (!gl_proc)
            return true;
        else
            return false;
    }

    bool slave_process() { return (!master_process()); }
};

#include "misc/misc.h"
#include "io/logout.h"
#include "io/perf.h"
