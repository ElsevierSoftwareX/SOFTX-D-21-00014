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
#include "primitives/matrix/matrix.h"

#include "blas/blas.h"
#include "comm/matvec_comm.h"

//////////

#ifndef XAMG_REVISION
#define XAMG_REVISION "Unspecified"
#endif

//////////

namespace XAMG {
namespace io {

// void print_version();

void print_bits(uint16_t ch);

void print_matrix_block(const matrix::matrix_block &block);
void print_matrix(const matrix::matrix &m, const std::string &str);

template <typename T>
void print_backend_matrix(T &mat);

/////////

void print_hline(const uint16_t nv);
void print_residuals_header(uint16_t res_type, const uint16_t nv);
void print_residuals_footer(const uint16_t nv);

template <typename T>
void print_residuals(const uint32_t iter, const vector::vector &res, const vector::vector &res0,
                     const vector::vector &conv);

/////////
// helpers:

static inline void sync(mpi::scope comm = mpi::GLOBAL) {
    if (!id.gl_proc)
        getchar();
    XAMG::mpi::barrier(comm);
}

/////////

static inline double timer() {
    struct timeval tp;
    struct timezone tzp;
    // int i;

    /*i = */ gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/////////

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
bool read_system(matrix::csr_matrix<F, I1, I2, I3, I4> &mat, vector::vector &x, vector::vector &b,
                 const std::string &path);

} // namespace io
} // namespace XAMG

#include "detail/print.inl"
#include "detail/reader.inl"
