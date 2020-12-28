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
struct Jacobi : public base_solver<F, NV> {
    const uint16_t nvecs = 2;
    const uint16_t comm_size = 0;
    DECLARE_INHERITED_FROM_BASESOLVER(Jacobi)
    virtual void init() override {
        auto &numa_layer = A.data_layer.find(segment::NUMA)->second;
        numa_layer.diag.get_inv_diag();

        base::init_base();
    }
};

template <typename F, uint16_t NV>
void Jacobi<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void Jacobi<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    assert(buffer.size() == (size_t)nvecs);
    vector::vector &x = *this->x;
    vector::vector &b = *this->b;

    uint16_t max_iters;
    uint16_t conv_info;
    uint16_t conv_check;
    float32_t relax_factor_f32;
    monitor.start("params");
    param_list.get_value("convergence_check", conv_check);
    param_list.get_value("max_iters", max_iters);
    param_list.get_value("convergence_info", conv_info);
    param_list.get_value("relax_factor", relax_factor_f32);
    monitor.stop("params");
    stats.reset(conv_check);
    auto &it = stats.iters;

    monitor.start("alloc");
    // inverted convergence flag; used to switch off updates to converged RHSs
    vector::vector iconv(conv);

    const vector::vector &a1 = blas::ConstVectorsCache<F>::get_ones_vec(NV);

    ////

    vector::vector rho0, res;
    rho0.alloc<F>(1, NV);
    res.alloc<F>(1, NV);

    vector::vector relax_conv;
    relax_conv.alloc<F>(1, NV);
    blas::set_const<F, NV>(relax_conv, (F)relax_factor_f32);
    monitor.stop("alloc");

    for (auto &buf : buffer)
        blas::set_const<F, NV>(buf, 0.0);

    vector::vector &r = buffer[0];
    const vector::vector &inv_diag = A.inv_diag();

    ///////////////////

    if (conv_check) {
        base::get_residual(x, r, rho0, conv_check);

        if (base::converged(rho0, rho0, iconv))
            return;

        do {
            ++it;
            blas::scal<F, NV>(inv_diag, r);
            blas::scal<F, NV>(iconv, relax_conv);
            blas::axpby<F, NV>(relax_conv, r, a1, x);

            base::get_residual(x, r, res, conv_check);

            if (base::converged(res, rho0, iconv))
                return;
        } while (it < max_iters);

        if (conv_info)
            io::print_residuals_footer(NV);
    } else {
        do {
            ++it;
            blas2::Ax_y<F, NV>(A, x, r, NV);
            blas::specific::merged_jacobi<F, NV>(x, b, inv_diag, r, relax_conv);
        } while (it < max_iters);
    }
}

} // namespace solver
} // namespace XAMG
