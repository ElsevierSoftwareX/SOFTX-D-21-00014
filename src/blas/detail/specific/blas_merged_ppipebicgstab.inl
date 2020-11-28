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
inline void merged_ppipebicgstab_group1(const vector::vector &r_, vector::vector &s_,
                                        vector::vector &p_, const vector::vector &w_,
                                        const vector::vector &z_, vector::vector &q_,
                                        const vector::vector &alpha, const vector::vector &beta,
                                        const vector::vector &omega) {
    //        4. \hat p_j = \hat r_j + beta_{j-1} (\hat p_{j-1} - omega_{j-1} \hat s_{j-1})
    //           \hat s_j = \hat w_j + beta_{j-1} (\hat s_{j-1} - omega_{j-1} \hat z_{j-1})
    //           \hat q_j = \hat r_j - \alpha_j \hat s_j
    r_.check(vector::vector::initialized);
    s_.check(vector::vector::initialized);
    p_.check(vector::vector::initialized);
    w_.check(vector::vector::initialized);
    z_.check(vector::vector::initialized);
    q_.check(vector::vector::allocated);
    alpha.check(vector::vector::initialized);
    beta.check(vector::vector::initialized);
    omega.check(vector::vector::initialized);

    const T *XAMG_RESTRICT r_ptr = r_.get_aligned_ptr<T>();
    T *XAMG_RESTRICT s_ptr = s_.get_aligned_ptr<T>();
    T *XAMG_RESTRICT p_ptr = p_.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT w_ptr = w_.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT z_ptr = z_.get_aligned_ptr<T>();
    T *XAMG_RESTRICT q_ptr = q_.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT beta_ptr = beta.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_ptr = omega.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    r_.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            p_ptr[i + nv] =
                r_ptr[i + nv] + beta_ptr[nv] * (p_ptr[i + nv] - omega_ptr[nv] * s_ptr[i + nv]);
            s_ptr[i + nv] =
                w_ptr[i + nv] + beta_ptr[nv] * (s_ptr[i + nv] - omega_ptr[nv] * z_ptr[i + nv]);
            q_ptr[i + nv] = r_ptr[i + nv] - alpha_ptr[nv] * s_ptr[i + nv];
        }
    }

    q_.if_initialized = true;

    p_.if_zero = false;
    s_.if_zero = false;
    q_.if_zero = false;

    perf.mem_read(5);
    perf.mem_write(3);
    perf.flop(10);
    XAMG_PERF_PRINT_DEBUG_INFO
}

template <typename T, const uint16_t NV>
inline void merged_ppipebicgstab_group2(const vector::vector &w, vector::vector &s,
                                        vector::vector &z, const vector::vector &t,
                                        const vector::vector &v, const vector::vector &r,
                                        vector::vector &q, vector::vector &y,
                                        const vector::vector &alpha, const vector::vector &beta,
                                        const vector::vector &omega, vector::vector &theta,
                                        vector::vector &phi, vector::vector &pi) {
    //    5. s_j = w_j + \beta_{j-1} (s_{j-1} - \omega_{j-1} z_{j-1})
    //       z_j = t_j + \beta_{j-1} (z_{j-1} - \omega_{j-1} v_{j-1})
    //       q_j = r_j - \alpha_j s_j
    //       y_j = w_j - \alpha_j z_j
    //       \theta_j = (q_j, y_j); \phi_j = (y_j, y_j), \pi = (q, q)
    w.check(vector::vector::initialized);
    s.check(vector::vector::initialized);
    z.check(vector::vector::initialized);
    t.check(vector::vector::initialized);
    v.check(vector::vector::initialized);
    r.check(vector::vector::initialized);
    q.check(vector::vector::allocated);
    y.check(vector::vector::allocated);
    alpha.check(vector::vector::initialized);
    beta.check(vector::vector::initialized);
    omega.check(vector::vector::initialized);
    theta.check(vector::vector::allocated);
    phi.check(vector::vector::allocated);
    pi.check(vector::vector::allocated);

    const T *XAMG_RESTRICT w_ptr = w.get_aligned_ptr<T>();
    T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();
    T *XAMG_RESTRICT z_ptr = z.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT t_ptr = t.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT v_ptr = v.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();
    T *XAMG_RESTRICT q_ptr = q.get_aligned_ptr<T>();
    T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT beta_ptr = beta.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_ptr = omega.get_aligned_ptr<T>();

    T *XAMG_RESTRICT theta_ptr = theta.get_aligned_ptr<T>();
    T *XAMG_RESTRICT phi_ptr = phi.get_aligned_ptr<T>();
    T *XAMG_RESTRICT pi_ptr = pi.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    w.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        theta_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        phi_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        pi_ptr[nv] = 0.0;

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            s_ptr[i + nv] =
                w_ptr[i + nv] + beta_ptr[nv] * (s_ptr[i + nv] - omega_ptr[nv] * z_ptr[i + nv]);
            z_ptr[i + nv] =
                t_ptr[i + nv] + beta_ptr[nv] * (z_ptr[i + nv] - omega_ptr[nv] * v_ptr[i + nv]);
            q_ptr[i + nv] = r_ptr[i + nv] - alpha_ptr[nv] * s_ptr[i + nv];
            y_ptr[i + nv] = w_ptr[i + nv] - alpha_ptr[nv] * z_ptr[i + nv];

            theta_ptr[nv] += q_ptr[i + nv] * y_ptr[i + nv];
            phi_ptr[nv] += y_ptr[i + nv] * y_ptr[i + nv];
            pi_ptr[nv] += q_ptr[i + nv] * q_ptr[i + nv];
        }
    }

    q.if_initialized = true;
    y.if_initialized = true;
    theta.if_initialized = true;
    phi.if_initialized = true;
    pi.if_initialized = true;

    s.if_zero = false;
    z.if_zero = false;
    q.if_zero = false;
    y.if_zero = false;
    theta.if_zero = false;
    phi.if_zero = false;
    pi.if_zero = false;

    perf.mem_read(6);
    perf.mem_write(4);
    perf.flop(18);
    XAMG_PERF_PRINT_DEBUG_INFO
}

template <typename T, const uint16_t NV>
inline void
merged_ppipebicgstab_group3(vector::vector &x, const vector::vector &p_, const vector::vector &q_,
                            vector::vector &r_, const vector::vector &w_, const vector::vector &z_,
                            const vector::vector &alpha_conv, const vector::vector &omega_conv,
                            const vector::vector &alpha, const vector::vector &omega) {
    //        13. x_{j+1} = x_{j} + \alpha_{j} \hat p_j + \omega_j \hat q_j
    //            \hat r_{j+1} = \hat q_{j} - \omega_{j} (\hat w_j - \alpha_j \hat z_j)
    x.check(vector::vector::initialized);
    p_.check(vector::vector::initialized);
    q_.check(vector::vector::initialized);
    r_.check(vector::vector::allocated);
    w_.check(vector::vector::initialized);
    z_.check(vector::vector::initialized);
    alpha.check(vector::vector::initialized);
    omega.check(vector::vector::initialized);
    alpha_conv.check(vector::vector::initialized);
    omega_conv.check(vector::vector::initialized);

    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT p_ptr = p_.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT q_ptr = q_.get_aligned_ptr<T>();
    T *XAMG_RESTRICT r_ptr = r_.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT w_ptr = w_.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT z_ptr = z_.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_conv_ptr = alpha_conv.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_conv_ptr = omega_conv.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_ptr = omega.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            x_ptr[i + nv] +=
                alpha_conv_ptr[nv] * p_ptr[i + nv] + omega_conv_ptr[nv] * q_ptr[i + nv];
            r_ptr[i + nv] =
                q_ptr[i + nv] - omega_ptr[nv] * (w_ptr[i + nv] - alpha_ptr[nv] * z_ptr[i + nv]);
        }
    }

    r_.if_initialized = true;

    x.if_zero = false;
    r_.if_zero = false;

    perf.mem_read(5);
    perf.mem_write(2);
    perf.flop(8);
    XAMG_PERF_PRINT_DEBUG_INFO
}

template <typename T, const uint16_t NV>
inline void
merged_ppipebicgstab_group4(vector::vector &r, const vector::vector &q, const vector::vector &y,
                            vector::vector &w, const vector::vector &t, const vector::vector &v,
                            const vector::vector &r0, const vector::vector &s,
                            const vector::vector &z, const vector::vector &alpha,
                            const vector::vector &omega, vector::vector &rho, vector::vector &sigma,
                            vector::vector &delta, vector::vector &psi) {
    //    14. r_j = q_j - \omega_j y_j
    //        w_{j+1} = y_{j} - \omega_{j} (t_j - \alpha_j v_j)
    //        \rho_{j+1} = (r_0, r_{j+1}), \sigma_j = (r_0, w_{j+1}),
    //        \delta_j = (r_0, s_{j}), \psi_j = (r_0, z_{j})
    r.check(vector::vector::allocated);
    q.check(vector::vector::initialized);
    y.check(vector::vector::initialized);
    w.check(vector::vector::allocated);
    t.check(vector::vector::initialized);
    v.check(vector::vector::initialized);
    r0.check(vector::vector::initialized);
    s.check(vector::vector::initialized);
    z.check(vector::vector::initialized);
    alpha.check(vector::vector::initialized);
    omega.check(vector::vector::initialized);
    rho.check(vector::vector::allocated);
    sigma.check(vector::vector::allocated);
    delta.check(vector::vector::allocated);
    psi.check(vector::vector::allocated);

    T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT q_ptr = q.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT y_ptr = y.get_aligned_ptr<T>();
    T *XAMG_RESTRICT w_ptr = w.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT t_ptr = t.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT v_ptr = v.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT r0_ptr = r0.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT z_ptr = z.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_ptr = omega.get_aligned_ptr<T>();

    T *XAMG_RESTRICT rho_ptr = rho.get_aligned_ptr<T>();
    T *XAMG_RESTRICT sigma_ptr = sigma.get_aligned_ptr<T>();
    T *XAMG_RESTRICT delta_ptr = delta.get_aligned_ptr<T>();
    T *XAMG_RESTRICT psi_ptr = psi.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    q.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        rho_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        sigma_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        delta_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        psi_ptr[nv] = 0.0;

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            r_ptr[i + nv] = q_ptr[i + nv] - omega_ptr[nv] * y_ptr[i + nv];
            w_ptr[i + nv] =
                y_ptr[i + nv] - omega_ptr[nv] * (t_ptr[i + nv] - alpha_ptr[nv] * v_ptr[i + nv]);

            rho_ptr[nv] += r0_ptr[i + nv] * r_ptr[i + nv];
            sigma_ptr[nv] += r0_ptr[i + nv] * w_ptr[i + nv];
            delta_ptr[nv] += r0_ptr[i + nv] * s_ptr[i + nv];
            psi_ptr[nv] += r0_ptr[i + nv] * z_ptr[i + nv];
        }
    }

    r.if_initialized = true;
    w.if_initialized = true;
    rho.if_initialized = true;
    sigma.if_initialized = true;
    delta.if_initialized = true;
    psi.if_initialized = true;

    r.if_zero = false;
    w.if_zero = false;
    rho.if_zero = false;
    sigma.if_zero = false;
    delta.if_zero = false;
    psi.if_zero = false;

    perf.mem_read(7);
    perf.mem_write(2);
    perf.flop(14);
    XAMG_PERF_PRINT_DEBUG_INFO
}

} // namespace specific
} // namespace blas
} // namespace XAMG
