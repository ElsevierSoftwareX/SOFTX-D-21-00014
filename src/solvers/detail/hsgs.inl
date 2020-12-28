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
struct HSGS : public base_solver<F, NV> {
    const uint16_t nvecs = 2;
    const uint16_t comm_size = 0;
    DECLARE_INHERITED_FROM_BASESOLVER(HSGS)
    virtual void init() {
        auto &numa_layer = A.data_layer.find(segment::NUMA)->second;
        numa_layer.diag.get_inv_diag();
        A.template construct_core_layer<F>();

        base::init_base();
    }
};

template <typename F, uint16_t NV>
void HSGS<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void HSGS<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    assert(buffer.size() == (size_t)nvecs);
    vector::vector &x = *this->x;
    vector::vector &b = *this->b;

    uint16_t conv_check;
    uint16_t max_iters;
    uint16_t conv_info;
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
    vector::vector &t = buffer[1];

    ///////////////////

    const vector::vector &inv_diag = A.inv_diag();

    if (conv_check) {
        base::get_residual(x, r, rho0);

        if (base::converged(rho0, rho0, iconv))
            return;
    }

    do {
        ++it;
        blas::scal<F, NV>(iconv, relax_conv);
        blas2::SGS<F, NV>(A, b, x, t, relax_conv, NV);

        if (conv_check) {
            base::get_residual(x, r, res);

            if (base::converged(res, rho0, iconv))
                return;
        }
    } while (it < max_iters);

    if ((conv_check) && (conv_info))
        io::print_residuals_footer(NV);
}
} // namespace solver
} // namespace XAMG
