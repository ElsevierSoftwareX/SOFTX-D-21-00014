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
struct MergedPipeBiCGStab : public base_solver<F, NV> {
    const uint16_t nvecs = 10;
    const uint16_t comm_size = 4;
    DECLARE_INHERITED_FROM_BASESOLVER(MergedPipeBiCGStab)
};

template <typename F, uint16_t NV>
void MergedPipeBiCGStab<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void MergedPipeBiCGStab<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    assert(buffer.size() == (size_t)nvecs);
    vector::vector &x = *this->x;
    vector::vector &b = *this->b;

    uint16_t conv_check;
    uint16_t max_iters;
    uint16_t conv_info;
    param_list.get_value("convergence_check", conv_check);
    param_list.get_value("max_iters", max_iters);
    param_list.get_value("convergence_info", conv_info);
    stats.reset(conv_check);
    auto &it = stats.iters;

    // inverted convergence flag; used to switch off updates to converged RHSs
    vector::vector iconv(conv);

    const vector::vector &a0 = blas::ConstVectorsCache<F>::get_zeroes_vec(NV);
    const vector::vector &a1 = blas::ConstVectorsCache<F>::get_ones_vec(NV);
    const vector::vector &a_1 = blas::ConstVectorsCache<F>::get_minus_ones_vec(NV);

    //////////

    vector::vector rho, rho0, res;
    rho.alloc<F>(1, NV);
    rho0.alloc<F>(1, NV);
    res.alloc<F>(1, NV);

    vector::vector scal1, pi, phi, psi, sigma, delta, alpha, beta, theta, omega;
    scal1.alloc<F>(1, NV);
    alpha.alloc<F>(1, NV);
    beta.alloc<F>(1, NV);
    omega.alloc<F>(1, NV);
    theta.alloc<F>(1, NV);
    pi.alloc<F>(1, NV);
    phi.alloc<F>(1, NV);
    psi.alloc<F>(1, NV);
    sigma.alloc<F>(1, NV);
    delta.alloc<F>(1, NV);

    vector::vector alpha_conv, omega_conv;
    alpha_conv.alloc<F>(1, NV);
    omega_conv.alloc<F>(1, NV);

    for (auto &buf : buffer)
        blas::set_const<F, NV>(buf, 0.0);

    vector::vector &r0 = buffer[0];
    vector::vector &r = buffer[1];
    vector::vector &w = buffer[2];
    vector::vector &t = buffer[3];
    vector::vector &p = buffer[4];
    vector::vector &s = buffer[5];
    vector::vector &z = buffer[6];
    vector::vector &q = buffer[7];
    vector::vector &y = buffer[8];
    vector::vector &v = buffer[9];

    ///////////////////

    //    1. r0 = b - Ax0; w0 = Ar0; t0 = Aw0
    //    2. \rho_0 = (r0, r0)
    base::get_residual(x, r0, rho0, conv_check);
    blas::copy<F, NV>(r0, r);

    if (base::converged(rho0, rho0, iconv))
        return;

    blas::copy<F, NV>(rho0, rho);

    blas2::Ax_y<F, NV>(A, r0, w, NV);
    blas::dot<F, NV>(w, r0, scal1);

    allreduce_buffer.push_vector(scal1);
    allreduce_buffer.init();
    allreduce_buffer.process_async_action();

    blas2::Ax_y<F, NV>(A, w, t, NV);

    allreduce_buffer.wait();
    allreduce_buffer.pull_vector(scal1);

    blas::xdivy_z<F, NV>(rho, scal1, alpha);
    //    alpha.print<F>("alpha0");

    ////////////////////

    perf.stop();

    do {
        ++it;
        if (it == 2)
            perf.start();

        if (it > 1) {
            blas::specific::merged_pipebicgstab_group1<F, NV>(p, r, s, w, z, t, v, q, y, beta,
                                                              omega, alpha, theta, phi, pi);
        } else {
            //            4. p_j = r_j
            //            5. s_j = w_j
            //            6. z_j = t_j
            //            7. q_j = r_j - \alpha_j s_j
            //            8. y_j = w_j - \alpha_j z_j
            //            9. \theta_j = (q_j, y_j); \phi_j = (y_j, y_j), \pi_j = (q_j, q_j)
            blas::specific::merged_pipebicgstab_group1a<F, NV>(p, r, s, w, z, t, v, q, y, beta,
                                                               omega, alpha, theta, phi, pi);
        }

        allreduce_buffer.push_vector(theta);
        allreduce_buffer.push_vector(phi);
        allreduce_buffer.push_vector(pi);
        allreduce_buffer.init();
        allreduce_buffer.process_async_action();

        //        10. v_j = A z_j
        blas2::Ax_y<F, NV>(A, z, v, NV, allreduce_buffer.get_token());

        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(pi);
        allreduce_buffer.pull_vector(phi);
        allreduce_buffer.pull_vector(theta);

        //        11. \omega = theta / phi
        blas::xdivy_z<F, NV>(theta, phi, omega);
        //        omega.print<F>("omega");

        //        12. x_{j+1} = x_{j} + \alpha_{j} p_j + \omega_j q_j
        //        13. r_j = q_j - \omega_j y_j
        blas::ax_y<F, NV>(iconv, alpha, alpha_conv);
        blas::ax_y<F, NV>(iconv, omega, omega_conv);
        blas::specific::merged_pipebicgstab_group2<F, NV>(x, p, q, r, y, alpha_conv, omega_conv,
                                                          alpha, omega);

        //        14. Convergence check
        blas::scal<F, NV>(a_1, omega);
        blas::axpby_z<F, NV>(a1, pi, omega, theta, res);
        blas::scal<F, NV>(a_1, omega);

        if (base::converged(res, rho0, iconv))
            return;

        //        17. w_{j+1} = y_{j} - \omega_{j} (t_j - \alpha_j v_j)
        //        18. \rho_{j+1} = (r_0, r_{j+1}), \sigma_j = (r_0, w_{j+1}),
        //            \delta_j = (r_0, s_{j}), \psi_j = (r_0, z_{j})
        blas::specific::merged_pipebicgstab_group3<F, NV>(w, y, t, v, r0, r, s, z, omega, alpha,
                                                          scal1, sigma, delta, psi);

        allreduce_buffer.push_vector(scal1);
        allreduce_buffer.push_vector(sigma);
        allreduce_buffer.push_vector(delta);
        allreduce_buffer.push_vector(psi);
        allreduce_buffer.init();
        allreduce_buffer.process_async_action();

        //        19. t_{j+1} = A w_{j+1}
        blas2::Ax_y<F, NV>(A, w, t, NV, allreduce_buffer.get_token());

        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(psi);
        allreduce_buffer.pull_vector(delta);
        allreduce_buffer.pull_vector(sigma);
        allreduce_buffer.pull_vector(scal1);

        //        20. \beta_{j} = (\alpha_j / \omega_j) (\rho_{j+1} / \rho_j)
        blas::xdivy_z<F, NV>(alpha, omega, beta);
        blas::xdivy_z<F, NV>(scal1, rho, alpha);
        blas::scal<F, NV>(alpha, beta);
        //        beta.print<F>("beta");

        blas::copy<F, NV>(scal1, rho);

        //        21. \alpha_{j+1} = \rho_{j+1} / (\sigma_j + \beta_j \delta_j - \beta_j \omega_j
        //        \psi_j)
        blas::scal<F, NV>(a_1, omega);
        blas::axpby_z<F, NV>(a1, delta, omega, psi, alpha);
        blas::scal<F, NV>(a_1, omega);
        blas::axpby_z<F, NV>(a1, sigma, beta, alpha, scal1);
        blas::xdivy_z<F, NV>(rho, scal1, alpha);
        //        alpha.print<F>("alpha");

        perf.stop();
    } while (it < max_iters);

    if ((conv_check) && (conv_info))
        io::print_residuals_footer(NV);
}

} // namespace solver
} // namespace XAMG
