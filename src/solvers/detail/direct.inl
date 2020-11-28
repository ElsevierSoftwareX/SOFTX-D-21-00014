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
namespace solver {

template <typename F, uint16_t NV>
struct Direct : public base_solver<F, NV> {
    const uint16_t nvecs = 0;
    const uint16_t comm_size = 0;
    matrix::matrix A_inv;
    DECLARE_INHERITED_FROM_BASESOLVER(Direct)
    virtual void init() override {
        matrix::inverse<F>(A, A_inv);
        A_inv.alloc_comm_buffers<F, NV>();
        base::init_base();
    }
};

template <typename F, uint16_t NV>
void Direct<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void Direct<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    assert(buffer.size() == (size_t)nvecs);
    vector::vector &x = *this->x;
    vector::vector &b = *this->b;

    blas2::Ax_y<F, NV>(A_inv, b, x, NV);
}

} // namespace solver
} // namespace XAMG
