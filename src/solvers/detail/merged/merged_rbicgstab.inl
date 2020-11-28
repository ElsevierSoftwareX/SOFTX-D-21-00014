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
struct MergedRBiCGStab : public base_solver<F, NV> {
    const uint16_t nvecs = 8;
    const uint16_t comm_size = 4;
    DECLARE_INHERITED_FROM_BASESOLVER(MergedRBiCGStab)
};

template <typename F, uint16_t NV>
void MergedRBiCGStab<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void MergedRBiCGStab<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
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

    vector::vector theta, phi, psi, delta, alpha, beta, omega, eta, temp;
    theta.alloc<F>(1, NV);
    phi.alloc<F>(1, NV);
    psi.alloc<F>(1, NV);
    delta.alloc<F>(1, NV);
    alpha.alloc<F>(1, NV);
    beta.alloc<F>(1, NV);
    omega.alloc<F>(1, NV);
    eta.alloc<F>(1, NV);
    temp.alloc<F>(1, NV);

    vector::vector alpha_conv, omega_conv;
    alpha_conv.alloc<F>(1, NV);
    omega_conv.alloc<F>(1, NV);

    for (auto &buf : buffer)
        blas::set_const<F, NV>(buf, 0.0);

    vector::vector &r0 = buffer[0];
    vector::vector &r = buffer[1];
    vector::vector &z = buffer[2];
    vector::vector &v_ = buffer[3];
    vector::vector &v = buffer[4];
    vector::vector &s = buffer[5];
    vector::vector &t_ = buffer[6];
    vector::vector &t = buffer[7];

    ///////////////////

    //    1. r0 = b - Ax0
    //    2. \rho_0 = (r0, r0)
    base::get_residual(x, r0, rho0);
    blas::copy<F, NV>(r0, r);

    if (base::converged(rho0, rho0, iconv))
        return;

    blas::copy<F, NV>(rho0, rho);

    //    3. z = M^{-1} r0
    blas::set_const<F, NV>(z, 0.0);
    precond->solve(z, r0);

    //    4. v_ = z
    blas::copy<F, NV>(z, v_);

    //    io::print_vector_norm<F, NV>(z, "res:");
    //    io::sync();

    //    io::print_vector_norm<F, NV>(x, " x0 ");
    //    io::print_vector_norm<F, NV>(b, " b0 ");

    ////////////////////

    perf.stop();

    do {
        ++it;
        if (it == 2)
            perf.start();

        //        6. v = A*v_
        blas2::Ax_y<F, NV>(A, v_, v, NV);
        //        io::print_vector_norm<F, NV>(v, "v");
        //        io::sync();

        //        7. \delta = (v, r0)
        blas::dot<F, NV>(v, r0, delta);
        allreduce_buffer.push_vector(delta);
        allreduce_buffer.init();
        allreduce_buffer.process_async_action();

        //        8. s = M^{-1} v
        blas::set_const<F, NV>(s, 0.0);
        precond->solve(s, v, allreduce_buffer.get_token());

        //        9. \alpha = \rho / \delta
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(delta);

        blas::xdivy_z<F, NV>(rho, delta, alpha);
        //        alpha.print<F>("alpha");

        //        10. t_ = z - \alpha s
        blas::scal<F, NV>(a_1, alpha);
        blas::axpby_z<F, NV>(a1, z, alpha, s, t_);
        blas::scal<F, NV>(a_1, alpha);

        //        11. t = A*t_
        blas2::Ax_y<F, NV>(A, t_, t, NV);

        //        12. r^ = r - \alpha v
        //        13. \theta = (t, r^), \phi = (t, t), \psi = (t, r0), \eta = (r^, r^)
        blas::specific::merged_rbicgstab_group1<F, NV>(r, v, t, r0, alpha, theta, phi, psi, eta);

        allreduce_buffer.push_vector(theta);
        allreduce_buffer.push_vector(phi);
        allreduce_buffer.push_vector(psi);
        allreduce_buffer.push_vector(eta);
        allreduce_buffer.init();
        allreduce_buffer.process_async_action();

        //        14. z = M^{-1} t
        blas::set_const<F, NV>(z, 0.0);
        precond->solve(z, t, allreduce_buffer.get_token());

        //        15. \omega = \theta / \phi
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(eta);
        allreduce_buffer.pull_vector(psi);
        allreduce_buffer.pull_vector(phi);
        allreduce_buffer.pull_vector(theta);

        blas::xdivy_z<F, NV>(theta, phi, omega);
        //        omega.print<F>("omega");

        //        16. r = r^ - \omega t
        blas::scal<F, NV>(a_1, omega);
        blas::axpby<F, NV>(omega, t, a1, r);
        blas::scal<F, NV>(a_1, omega);

        //        17. Convergence check
        blas::ax_y<F, NV>(omega, theta, res);
        blas::axpby<F, NV>(a1, eta, a_1, res);

        blas::ax_y<F, NV>(iconv, alpha, alpha_conv);
        blas::ax_y<F, NV>(iconv, omega, omega_conv);

        if (base::converged(res, rho0, iconv)) {
            //            18. x = x + \alpha v_  + \omega t_
            blas::axpbypcz<F, NV>(alpha_conv, v_, omega_conv, t_, a1, x);
            return;
        }

        //        21. \rho_n = - \omega \psi
        blas::ax_y<F, NV>(omega, psi, rho);
        blas::scal<F, NV>(a_1, rho);

        //        22. \beta = rho_{j+1} * alpha / (rho_{j} * omega)
        blas::ax_y<F, NV>(delta, omega, temp);
        blas::xdivy_z<F, NV>(rho, temp, beta);
        //        beta.print<F>("beta");

        //        23. x = x + \alpha v_  + \omega t_
        //        24. z = t_ - \omega z
        //        25. v_ = z + beta v_ - beta*omega s
        //        blas::specific::merged_rbicgstab_group2<F, NV>(x, v_, t_, z, s, alpha, omega,
        //        beta);
        blas::ax_y<F, NV>(iconv, alpha, alpha_conv);
        blas::ax_y<F, NV>(iconv, omega, omega_conv);
        blas::specific::merged_rbicgstab_group2<F, NV>(x, v_, t_, z, s, alpha_conv, omega_conv,
                                                       alpha, omega, beta);

        perf.stop();
    } while (it < max_iters);

    if ((conv_check) && (conv_info))
        io::print_residuals_footer(NV);
}

} // namespace solver
} // namespace XAMG
