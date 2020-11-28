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
struct IBiCGStab : public base_solver<F, NV> {
    const uint16_t nvecs = 9;
    const uint16_t comm_size = 7;
    DECLARE_INHERITED_FROM_BASESOLVER(IBiCGStab)
};

template <typename F, uint16_t NV>
void IBiCGStab<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void IBiCGStab<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
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

    vector::vector rho, rho0, rho1, res;
    rho.alloc<F>(1, NV);
    rho0.alloc<F>(1, NV);
    rho1.alloc<F>(1, NV);
    res.alloc<F>(1, NV);

    vector::vector scal1, scal2, scal3, scal4, alpha, alpha_1, beta, omega;
    vector::vector sigma_1, pi, tau, sigma, phi, theta, gamma, eta, kappa, nu, delta;
    scal1.alloc<F>(1, NV);
    scal2.alloc<F>(1, NV);
    scal3.alloc<F>(1, NV);
    scal4.alloc<F>(1, NV);
    alpha.alloc<F>(1, NV);
    alpha_1.alloc<F>(1, NV);
    beta.alloc<F>(1, NV);
    omega.alloc<F>(1, NV);

    sigma_1.alloc<F>(1, NV);
    pi.alloc<F>(1, NV);
    tau.alloc<F>(1, NV);
    sigma.alloc<F>(1, NV);
    phi.alloc<F>(1, NV);
    theta.alloc<F>(1, NV);
    gamma.alloc<F>(1, NV);
    eta.alloc<F>(1, NV);
    kappa.alloc<F>(1, NV);
    nu.alloc<F>(1, NV);
    delta.alloc<F>(1, NV);

    vector::vector a1_conv, omega_conv;
    a1_conv.alloc<F>(1, NV);
    omega_conv.alloc<F>(1, NV);
    blas::set_const<F, NV>(a1_conv, 1.0);

    for (auto &buf : buffer)
        blas::set_const<F, NV>(buf, 0.0);

    vector::vector &r0 = buffer[0];
    vector::vector &r = buffer[1];
    vector::vector &u = buffer[2];
    vector::vector &f0 = buffer[3];
    vector::vector &q = buffer[4];
    vector::vector &v = buffer[5];
    vector::vector &z = buffer[6];
    vector::vector &s = buffer[7];
    vector::vector &t = buffer[8];

    ///////////////////

    //    1. r0 = b - Ax0, u0 = Ar0, f0 = A^Tr0, q_0 = v_0 = z_0 = 0
    base::get_residual(x, r0, rho0);
    blas::copy<F, NV>(r0, r);

    if (base::converged(rho0, rho0, iconv))
        return;

    blas2::Ax_y<F, NV>(A, r0, u, NV);
    blas::copy<F, NV>(u, f0);

    blas::set_const<F, NV>(q, a0);
    blas::set_const<F, NV>(v, a0);
    blas::set_const<F, NV>(z, a0);

    //    2. sigma_1 = pi = tau = 0, sigma = (r_0, u_0), rho_0 = alpha = omega = 1, phi = (r0, r0)
    blas::set_const<F, NV>(sigma_1, a0);
    blas::set_const<F, NV>(pi, a0);
    blas::set_const<F, NV>(tau, a0);

    blas::copy<F, NV>(rho0, phi);
    blas::set_const<F, NV>(rho1, a1);
    blas::set_const<F, NV>(alpha, a1);
    blas::set_const<F, NV>(omega, a1);

    blas::dot<F, NV>(r0, u, sigma);

    allreduce_buffer.push_vector(sigma);
    allreduce_buffer.init();
    allreduce_buffer.process_sync_action();
    allreduce_buffer.wait();
    allreduce_buffer.pull_vector(sigma);

    //    io::print_vector_norm<F, NV>(x, " x0 ");
    //    io::print_vector_norm<F, NV>(b, " b0 ");

    ////////////////////

    perf.stop();

    do {
        ++it;
        if (it == 2)
            perf.start();

        blas::copy<F, NV>(rho1, rho);

        //        4. rho{j+1} = phi_j - omega_j sigma_{j-1} + omega_j alpha_j pi_j
        blas::scal<F, NV>(a_1, alpha);
        blas::axpby_z<F, NV>(a1, sigma_1, alpha, pi, rho1);
        blas::scal<F, NV>(a_1, alpha);
        blas::scal<F, NV>(a_1, omega);
        blas::axpby<F, NV>(a1, phi, omega, rho1);
        blas::scal<F, NV>(a_1, omega);

        //        5. delta_{j+1} = rho_{j+1} * alpha_j / rho_j, beta_{j+1} = delta_{j+1} / omega_j
        blas::xdivy_z<F, NV>(rho1, rho, delta);
        blas::scal<F, NV>(alpha, delta);

        blas::xdivy_z<F, NV>(delta, omega, beta);
        //        beta.print<F>("beta");

        //        6. tau_{j+1} = sigma_j + beta_{j+1} tau_j - delta_{j+1} pi_j
        blas::scal<F, NV>(a_1, delta);
        blas::axpbypcz<F, NV>(a1, sigma, delta, pi, beta, tau);
        blas::scal<F, NV>(a_1, delta);
        //        io::print_vector_norm<F, NV>(tau, " tau ");

        //        7. alpha_{j+1} = \frac{\rho_{j+1}}{\tau_{j+1}}
        blas::copy<F, NV>(alpha, alpha_1);
        blas::xdivy_z<F, NV>(rho1, tau, alpha);
        //        io::print_vector_norm<F, NV>(alpha, " alpha ");

        //        8. z_{j+1} = \alpha_{j+1} r_j + \beta_{j+1} \frac{\alpha_{j+1}}{\alpha_j} z_j -
        //        \alpha_{j+1} \delta_{j+1} v_j
        blas::xdivy_z<F, NV>(alpha, alpha_1, scal1);
        blas::scal<F, NV>(beta, scal1);

        blas::ax_y<F, NV>(alpha, delta, scal2);
        blas::scal<F, NV>(a_1, scal2);

        blas::axpbypcz<F, NV>(alpha, r, scal2, v, scal1, z);

        //        9.  v_{j+1} = u_j + beta_{j+1} v_j - delta_{j+1} q_j
        blas::scal<F, NV>(a_1, delta);
        blas::axpbypcz<F, NV>(a1, u, delta, q, beta, v);
        blas::scal<F, NV>(a_1, delta);

        //        10. s_{j+1} = r_j - alpha_{j+1} v_{j+1}
        blas::scal<F, NV>(a_1, alpha);
        blas::axpby_z<F, NV>(a1, r, alpha, v, s);
        blas::scal<F, NV>(a_1, alpha);

        //        11. q_{j+1} = A v_{j+1}
        blas2::Ax_y<F, NV>(A, v, q, NV);
        //        v.print<F>("v");
        //        io::print_vector_norm<F, NV>(q, "q");
        //        io::sync();

        //        12. t_{j+1} = u_j - \alpha_{j+1} q_{j+1}
        blas::scal<F, NV>(a_1, alpha);
        blas::axpby_z<F, NV>(a1, u, alpha, q, t);
        blas::scal<F, NV>(a_1, alpha);

        //        13. phi_{j+1} = (r_0, s_{j+1})
        blas::dot<F, NV>(r0, s, phi);

        //        14. pi_{j+1} = (r_0, q_{j+1})
        blas::dot<F, NV>(r0, q, pi);

        //        15. gamma_{j+1} = (f_0, s_{j+1})
        blas::dot<F, NV>(f0, s, gamma);

        //        16. eta_{j+1} = (f_0, t_{j+1})
        blas::dot<F, NV>(f0, t, eta);

        //        17. theta_{j+1} = (s_{j+1}, t_{j+1})
        blas::dot<F, NV>(s, t, theta);

        //        18. kappa_{j+1} = (t_{j+1}, t_{j+1})
        blas::dot<F, NV>(t, t, kappa);

        //        19. nu_{j+1} = (s_{j+1}, s_{j+1})
        blas::dot<F, NV>(s, s, nu);

        allreduce_buffer.push_vector(phi);
        allreduce_buffer.push_vector(pi);
        allreduce_buffer.push_vector(gamma);
        allreduce_buffer.push_vector(eta);
        allreduce_buffer.push_vector(theta);
        allreduce_buffer.push_vector(kappa);
        allreduce_buffer.push_vector(nu);
        allreduce_buffer.init();
        allreduce_buffer.process_sync_action();
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(nu);
        allreduce_buffer.pull_vector(kappa);
        allreduce_buffer.pull_vector(theta);
        allreduce_buffer.pull_vector(eta);
        allreduce_buffer.pull_vector(gamma);
        allreduce_buffer.pull_vector(pi);
        allreduce_buffer.pull_vector(phi);

        //        phi.print<F>("phi");

        //        20. omega_{j+1} = theta_{j+1} / kappa_{j+1}
        blas::xdivy_z<F, NV>(theta, kappa, omega);
        //        io::print_vector_norm<F, NV>(omega, "omega");
        //        omega.print<F>("omega");

        //        21. sigma_{j+1} = gamma_{j+1} - omega_{j+1} eta_{j+1}
        blas::copy<F, NV>(sigma, sigma_1);

        blas::scal<F, NV>(a_1, omega);
        blas::axpby_z<F, NV>(a1, gamma, omega, eta, sigma);
        blas::scal<F, NV>(a_1, omega);
        //        io::print_vector_norm<F, NV>(sigma, "sigma");

        //        22. r_{j+1} = s_{j+1} - \omega_{j+1} t_{j+1}
        blas::scal<F, NV>(a_1, omega);
        blas::axpby_z<F, NV>(a1, s, omega, t, r);
        blas::scal<F, NV>(a_1, omega);
        //        io::print_vector_norm<F, NV>(r, "r");

        //        23. x_{j+1} = x_j + z_{j+1} + \omega_{j+1} s_{j+1}
        //        blas::axpbypcz<F, NV>(a1, z, omega, s, a1, x);
        blas::ax_y<F, NV>(iconv, a1, a1_conv);
        blas::ax_y<F, NV>(iconv, omega, omega_conv);
        blas::axpbypcz<F, NV>(a1_conv, z, omega_conv, s, a1, x);

        //        24. Convergence check
        blas::scal<F, NV>(a_1, omega);
        blas::axpby_z<F, NV>(a1, nu, omega, theta, res);
        blas::scal<F, NV>(a_1, omega);
        //        res.print<F>("res");
        //        io::print_vector_norm<F, NV>(res, "res");

        if (base::converged(res, rho0, iconv))
            return;

        //        15. u_{j+1} = A r_{j+1}
        blas2::Ax_y<F, NV>(A, r, u, NV);

        perf.stop();
    } while (it < max_iters);

    if ((conv_check) && (conv_info))
        io::print_residuals_footer(NV);
}

} // namespace solver
} // namespace XAMG
