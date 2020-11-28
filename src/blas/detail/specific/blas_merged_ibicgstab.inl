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
namespace blas {
namespace specific {

template <typename T, const uint16_t NV>
inline void merged_ibicgstab_group1(const vector::vector &r, vector::vector &z, vector::vector &v,
                                    const vector::vector &u, const vector::vector &q,
                                    vector::vector &s, const vector::vector &alpha,
                                    const vector::vector &alpha_1, const vector::vector &beta,
                                    const vector::vector &delta) {
    //    8. z_{j+1} = alpha_{j+1} r_j + (beta_{j+1} * alpha_{j+1} / alpha_j) * z_j - alpha_{j+1} * delta_{j+1} v_j
    //    9. v_{j+1} = u_j + beta_{j+1} v_j - delta_{j+1} q_j
    //    10. s_{j+1} = r_j - alpha_{j+1} v_{j+1}
    r.check(vector::vector::initialized);
    z.check(vector::vector::initialized);
    v.check(vector::vector::initialized);
    u.check(vector::vector::initialized);
    q.check(vector::vector::initialized);
    s.check(vector::vector::allocated);
    alpha.check(vector::vector::initialized);
    alpha_1.check(vector::vector::initialized);
    beta.check(vector::vector::initialized);
    delta.check(vector::vector::initialized);

    if (r.if_zero)
        blas::forced_set_const<T, NV>(r, 0.0);
    if (z.if_zero)
        blas::set_const<T, NV>(z, 0.0, true);
    if (v.if_zero)
        blas::set_const<T, NV>(v, 0.0, true);
    if (u.if_zero)
        blas::forced_set_const<T, NV>(u, 0.0);
    if (q.if_zero)
        blas::forced_set_const<T, NV>(q, 0.0);
    //    if (s.if_zero)
    //        blas::set_const<T, NV>(s, 0.0);

    if (alpha.if_zero)
        blas::forced_set_const<T, NV>(alpha, 0.0);
    if (alpha_1.if_zero)
        blas::forced_set_const<T, NV>(alpha_1, 0.0);
    if (beta.if_zero)
        blas::forced_set_const<T, NV>(beta, 0.0);
    if (delta.if_zero)
        blas::forced_set_const<T, NV>(delta, 0.0);

    const T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();
    T *XAMG_RESTRICT z_ptr = z.get_aligned_ptr<T>();
    T *XAMG_RESTRICT v_ptr = v.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT u_ptr = u.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT q_ptr = q.get_aligned_ptr<T>();
    T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT alpha_1_ptr = alpha_1.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT beta_ptr = beta.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT delta_ptr = delta.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    r.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            z_ptr[i + nv] = alpha_ptr[nv] * r_ptr[i + nv] +
                            (beta_ptr[nv] * alpha_ptr[nv] / alpha_1_ptr[nv]) * z_ptr[i + nv] -
                            alpha_ptr[nv] * delta_ptr[nv] * v_ptr[i + nv];

            v_ptr[i + nv] =
                u_ptr[i + nv] + beta_ptr[nv] * v_ptr[i + nv] - delta_ptr[nv] * q_ptr[i + nv];
            s_ptr[i + nv] = r_ptr[i + nv] - alpha_ptr[nv] * v_ptr[i + nv];
        }
    }

    s.if_initialized = true;
    z.if_zero = false;
    v.if_zero = false;
    s.if_zero = false;

    perf.mem_read(5);
    perf.mem_write(3);
    perf.flop(11);
    XAMG_PERF_PRINT_DEBUG_INFO
}

template <typename T, const uint16_t NV>
inline void
merged_ibicgstab_group2(vector::vector &t, const vector::vector &u, const vector::vector &q,
                        const vector::vector &r0, const vector::vector &s, const vector::vector &f0,
                        const vector::vector &alpha, vector::vector &phi, vector::vector &pi,
                        vector::vector &gamma, vector::vector &eta, vector::vector &theta,
                        vector::vector &kappa, vector::vector &nu) {
    //    12. t_{j+1} = u_j - alpha_{j+1} q_{j+1}
    //    13. phi_{j+1} = (r_0, s_{j+1})
    //    14. pi_{j+1} = (r_0, q_{j+1})
    //    15. gamma_{j+1} = (f_0, s_{j+1})
    //    16. eta_{j+1} = (f_0, t_{j+1})
    //    17. theta_{j+1} = (s_{j+1}, t_{j+1})
    //    18. kappa_{j+1} = (t_{j+1}, t_{j+1})
    //    19. nu_{j+1} = (s_{j+1}, s_{j+1})
    t.check(vector::vector::allocated);
    u.check(vector::vector::initialized);
    q.check(vector::vector::initialized);
    r0.check(vector::vector::initialized);
    s.check(vector::vector::initialized);
    f0.check(vector::vector::initialized);
    alpha.check(vector::vector::initialized);
    phi.check(vector::vector::allocated);
    pi.check(vector::vector::allocated);
    gamma.check(vector::vector::allocated);
    eta.check(vector::vector::allocated);
    theta.check(vector::vector::allocated);
    kappa.check(vector::vector::allocated);
    nu.check(vector::vector::allocated);

    if (u.if_zero)
        blas::forced_set_const<T, NV>(u, 0.0);
    if (q.if_zero)
        blas::forced_set_const<T, NV>(q, 0.0);
    if (r0.if_zero)
        blas::forced_set_const<T, NV>(r0, 0.0);
    if (s.if_zero)
        blas::forced_set_const<T, NV>(s, 0.0);
    if (f0.if_zero)
        blas::forced_set_const<T, NV>(f0, 0.0);

    if (alpha.if_zero)
        blas::forced_set_const<T, NV>(alpha, 0.0);

    T *XAMG_RESTRICT t_ptr = t.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT u_ptr = u.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT q_ptr = q.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT r0_ptr = r0.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT f0_ptr = f0.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    T *XAMG_RESTRICT phi_ptr = phi.get_aligned_ptr<T>();
    T *XAMG_RESTRICT pi_ptr = pi.get_aligned_ptr<T>();
    T *XAMG_RESTRICT gamma_ptr = gamma.get_aligned_ptr<T>();
    T *XAMG_RESTRICT eta_ptr = eta.get_aligned_ptr<T>();
    T *XAMG_RESTRICT theta_ptr = theta.get_aligned_ptr<T>();
    T *XAMG_RESTRICT kappa_ptr = kappa.get_aligned_ptr<T>();
    T *XAMG_RESTRICT nu_ptr = nu.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    u.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        phi_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        pi_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        gamma_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        eta_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        theta_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        kappa_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        nu_ptr[nv] = 0.0;

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            t_ptr[i + nv] = u_ptr[i + nv] - alpha_ptr[nv] * q_ptr[i + nv];

            phi_ptr[nv] += r0_ptr[i + nv] * s_ptr[i + nv];
            pi_ptr[nv] += r0_ptr[i + nv] * q_ptr[i + nv];
            gamma_ptr[nv] += f0_ptr[i + nv] * s_ptr[i + nv];
            eta_ptr[nv] += f0_ptr[i + nv] * t_ptr[i + nv];
            theta_ptr[nv] += s_ptr[i + nv] * t_ptr[i + nv];
            kappa_ptr[nv] += t_ptr[i + nv] * t_ptr[i + nv];
            nu_ptr[nv] += s_ptr[i + nv] * s_ptr[i + nv];
        }
    }

    t.if_initialized = true;
    phi.if_initialized = true;
    pi.if_initialized = true;
    gamma.if_initialized = true;
    eta.if_initialized = true;
    theta.if_initialized = true;
    kappa.if_initialized = true;
    nu.if_initialized = true;
    t.if_zero = false;
    phi.if_zero = false;
    pi.if_zero = false;
    gamma.if_zero = false;
    eta.if_zero = false;
    theta.if_zero = false;
    kappa.if_zero = false;
    nu.if_zero = false;

    perf.mem_read(5);
    perf.mem_write(1);
    perf.flop(16);
    XAMG_PERF_PRINT_DEBUG_INFO
}

template <typename T, const uint16_t NV>
inline void merged_ibicgstab_group3(vector::vector &r, const vector::vector &s,
                                    const vector::vector &t, vector::vector &x,
                                    const vector::vector &z, const vector::vector &a1_conv,
                                    const vector::vector &omega_conv, const vector::vector &omega) {
    //    22. r_{j+1} = s_{j+1} - \omega_{j+1} t_{j+1}
    //    23. x_{j+1} = x_j + z_{j+1} + \omega_{j+1} s_{j+1}
    r.check(vector::vector::allocated);
    s.check(vector::vector::initialized);
    t.check(vector::vector::initialized);
    x.check(vector::vector::initialized);
    z.check(vector::vector::initialized);
    omega.check(vector::vector::initialized);
    a1_conv.check(vector::vector::initialized);
    omega_conv.check(vector::vector::initialized);

    if (s.if_zero)
        blas::forced_set_const<T, NV>(s, 0.0);
    if (t.if_zero)
        blas::forced_set_const<T, NV>(t, 0.0);
    if (x.if_zero)
        blas::set_const<T, NV>(x, 0.0, true);
    if (z.if_zero)
        blas::forced_set_const<T, NV>(z, 0.0);

    if (omega.if_zero)
        blas::forced_set_const<T, NV>(omega, 0.0);

    T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT t_ptr = t.get_aligned_ptr<T>();
    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT z_ptr = z.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT a1_conv_ptr = a1_conv.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_conv_ptr = omega_conv.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_ptr = omega.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    s.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            r_ptr[i + nv] = s_ptr[i + nv] - omega_ptr[nv] * t_ptr[i + nv];
            x_ptr[i + nv] += a1_conv_ptr[nv] * z_ptr[i + nv] + omega_conv_ptr[nv] * s_ptr[i + nv];
        }
    }

    r.if_initialized = true;
    r.if_zero = false;
    x.if_zero = false;

    perf.mem_read(4);
    perf.mem_write(2);
    perf.flop(5);
    XAMG_PERF_PRINT_DEBUG_INFO
}

} // namespace specific
} // namespace blas
} // namespace XAMG
