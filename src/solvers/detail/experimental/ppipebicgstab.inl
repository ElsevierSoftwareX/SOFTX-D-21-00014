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

#pragma once

namespace XAMG {
namespace solver {

template <typename F, uint16_t NV>
struct PPipeBiCGStab : public base_solver<F, NV> {
    const uint16_t nvecs = 15;
    const uint16_t comm_size = 4;
    DECLARE_INHERITED_FROM_BASESOLVER(PPipeBiCGStab)
};

template <typename F, uint16_t NV>
void PPipeBiCGStab<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void PPipeBiCGStab<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
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

    vector::vector scal1, scal2, pi, phi, psi, sigma, delta, alpha, beta, theta, omega;
    scal1.alloc<F>(1, NV);
    scal2.alloc<F>(1, NV);
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
    vector::vector &r_ = buffer[1];
    vector::vector &r = buffer[2];
    vector::vector &w_ = buffer[3];
    vector::vector &w = buffer[4];
    vector::vector &p_ = buffer[5];
    vector::vector &s_ = buffer[6];
    vector::vector &s = buffer[7];
    vector::vector &z_ = buffer[8];
    vector::vector &z = buffer[9];
    vector::vector &q_ = buffer[10];
    vector::vector &q = buffer[11];
    vector::vector &y = buffer[12];
    vector::vector &t = buffer[13];
    vector::vector &v = buffer[14];

    ///////////////////

    //    1. r0 = b - Ax0; r_ = M^{-1}r0; w0 = Ar_; w_ = M^{-1}w0; t0 = Aw_
    //    2. rho_0 = (r0, r0); alpha_0 = rho_0 / (r_0,w_0); beta_{-1} = 0
    base::get_residual(x, r0, rho0, conv_check);
    blas::copy<F, NV>(r0, r);

    if (base::converged(rho0, rho0, iconv))
        return;

    //    precond
    blas::set_const<F, NV>(r_, 0.0);
    precond->solve(r_, r0);
    blas2::Ax_y<F, NV>(A, r_, w, NV);

    //    precond
    blas::set_const<F, NV>(w_, 0.0);
    precond->solve(w_, w);
    blas2::Ax_y<F, NV>(A, w_, t, NV);

    blas::copy<F, NV>(rho0, rho);

    blas::dot<F, NV>(w, r0, scal1);

    allreduce_buffer.push_vector(scal1);
    allreduce_buffer.init();
    allreduce_buffer.process_sync_action();
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
            //        io::print_vector_norm<F, NV>(w_, "w_");
            //        io::print_vector_norm<F, NV>(z_, "z_");

            blas::ax_y<F, NV>(omega, beta, scal1);
            blas::scal<F, NV>(a_1, scal1);

            //            4. \hat p_j = \hat r_j + beta_{j-1} (\hat p_{j-1} - omega_{j-1} \hat
            //            s_{j-1})
            blas::axpbypcz<F, NV>(a1, r_, scal1, s_, beta, p_);

            //            5. s_j = w_j + \beta_{j-1} (s_{j-1} - \omega_{j-1} z_{j-1})
            blas::axpbypcz<F, NV>(a1, w, scal1, z, beta, s);

            //            6. \hat s_j = \hat w_j + beta_{j-1} (\hat s_{j-1} - omega_{j-1} \hat
            //            z_{j-1})
            blas::axpbypcz<F, NV>(a1, w_, scal1, z_, beta, s_);

            //            7. z_j = t_j + \beta_{j-1} (z_{j-1} - \omega_{j-1} v_{j-1})
            blas::axpbypcz<F, NV>(a1, t, scal1, v, beta, z);
        } else {
            //            4. \hat p_j = \hat r_j
            blas::copy<F, NV>(r_, p_);
            //            5. s_j = w_j
            blas::copy<F, NV>(w, s);
            //            6. \hat s_j = \hat w_j
            blas::copy<F, NV>(w_, s_);
            //            7. z_j = t_j
            blas::copy<F, NV>(t, z);
        }
        //        io::print_vector_norm<F, NV>(p_, "p_");
        //        io::print_vector_norm<F, NV>(s_, "s_");

        blas::ax_y<F, NV>(a_1, alpha, scal1);

        //        8. q_j = r_j - \alpha_j s_j
        blas::axpby_z<F, NV>(a1, r, scal1, s, q);

        //        9. \hat q_j = \hat r_j - \alpha_j \hat s_j
        blas::axpby_z<F, NV>(a1, r_, scal1, s_, q_);

        //        10. y_j = w_j - \alpha_j z_j
        blas::axpby_z<F, NV>(a1, w, scal1, z, y);

        //        11. \theta_j = (q_j, y_j); \phi_j = (y_j, y_j),
        blas::dot<F, NV>(q, y, theta);
        blas::dot<F, NV>(y, y, phi);
        blas::dot<F, NV>(q, q, pi);
        allreduce_buffer.push_vector(theta);
        allreduce_buffer.push_vector(phi);
        allreduce_buffer.push_vector(pi);
        allreduce_buffer.init();
        allreduce_buffer.process_async_action();

        //        12. \hat z = M^{-1} z; v_j = A \hat z
        blas::set_const<F, NV>(z_, 0.0);
        precond->solve(z_, z, allreduce_buffer.get_token());
        blas2::Ax_y<F, NV>(A, z_, v, NV, allreduce_buffer.get_token());

        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(pi);
        allreduce_buffer.pull_vector(phi);
        allreduce_buffer.pull_vector(theta);

        //        13. \omega = theta / phi
        blas::xdivy_z<F, NV>(theta, phi, omega);
        //        omega.print<F>("omega");

        //        14. x_{j+1} = x_{j} + \alpha_{j} \hat p_j + \omega_j \hat q_j
        //        blas::axpbypcz<F, NV>(alpha, p_, omega, q_, a1, x);
        blas::ax_y<F, NV>(iconv, alpha, alpha_conv);
        blas::ax_y<F, NV>(iconv, omega, omega_conv);
        blas::axpbypcz<F, NV>(alpha_conv, p_, omega_conv, q_, a1, x);

        //        15. r_j = q_j - \omega_j y_j
        blas::ax_y<F, NV>(a_1, omega, scal1);
        blas::axpby_z<F, NV>(a1, q, scal1, y, r);

        //        16. Convergence check
        blas::axpby_z<F, NV>(a1, pi, scal1, theta, res);

        if (base::converged(res, rho0, iconv))
            return;

        //        19. \hat r_{j+1} = \hat q_{j} - \omega_{j} (\hat w_j - \alpha_j \hat z_j)
        blas::ax_y<F, NV>(a_1, alpha, scal1);
        blas::axpby_z<F, NV>(a1, w_, scal1, z_, r_);
        blas::ax_y<F, NV>(a_1, omega, scal1);
        blas::axpby<F, NV>(a1, q_, scal1, r_);

        //        20. w_{j+1} = y_{j} - \omega_{j} (t_j - \alpha_j v_j)
        blas::ax_y<F, NV>(a_1, alpha, scal1);
        blas::axpby_z<F, NV>(a1, t, scal1, v, w);
        blas::ax_y<F, NV>(a_1, omega, scal1);
        blas::axpby<F, NV>(a1, y, scal1, w);

        //        21. \rho_{j+1} = (r_0, r_{j+1}), \sigma_j = (r_0, w_{j+1}),
        //            \delta_j = (r_0, s_{j}), \psi_j = (r_0, z_{j})
        blas::dot<F, NV>(r0, r, scal1);
        blas::dot<F, NV>(r0, w, sigma);
        blas::dot<F, NV>(r0, s, delta);
        blas::dot<F, NV>(r0, z, psi);
        allreduce_buffer.push_vector(scal1);
        allreduce_buffer.push_vector(sigma);
        allreduce_buffer.push_vector(delta);
        allreduce_buffer.push_vector(psi);
        allreduce_buffer.init();
        allreduce_buffer.process_async_action();

        //        22. \hat w = M^{-1} w; t_{j+1} = A \hat w
        blas::set_const<F, NV>(w_, 0.0);
        precond->solve(w_, w, allreduce_buffer.get_token());
        blas2::Ax_y<F, NV>(A, w_, t, NV, allreduce_buffer.get_token());

        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(psi);
        allreduce_buffer.pull_vector(delta);
        allreduce_buffer.pull_vector(sigma);
        allreduce_buffer.pull_vector(scal1);

        //        23. \beta_{j} = (\alpha_j / \omega_j) (\rho_{j+1} / \rho_j)
        blas::xdivy_z<F, NV>(scal1, rho, scal2);
        blas::xdivy_z<F, NV>(alpha, omega, beta);
        blas::scal<F, NV>(scal2, beta);
        //        beta.print<F>("beta");

        blas::copy<F, NV>(scal1, rho);

        //        24. \alpha_{j+1} = \rho_{j+1} / (\sigma_j + \beta_j \delta_j - \beta_j \omega_j
        //        \psi_j)
        for (uint16_t nv = 0; nv < NV; ++nv)
            blas::ax_y<F, NV>(a_1, omega, scal1);
        blas::axpby_z<F, NV>(a1, delta, scal1, psi, scal2);
        blas::axpby_z<F, NV>(a1, sigma, beta, scal2, scal1);
        blas::xdivy_z<F, NV>(rho, scal1, alpha);
        //        alpha.print<F>("alpha");

        perf.stop();
    } while (it < max_iters);

    if ((conv_check) && (conv_info))
        io::print_residuals_footer(NV);
}

} // namespace solver
} // namespace XAMG
