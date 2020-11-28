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
inline void merged_rbicgstab_group1(vector::vector &r, const vector::vector &v,
                                    const vector::vector &t, const vector::vector &r0,
                                    const vector::vector &alpha, vector::vector &theta,
                                    vector::vector &phi, vector::vector &psi, vector::vector &eta) {
    //    12. r^ = r - \alpha v
    //    13. \theta = (t, r^), \phi = (t, t), \psi = (t, r0), \eta = (r^, r^)
    r.check(vector::vector::initialized);
    v.check(vector::vector::initialized);
    t.check(vector::vector::initialized);
    r0.check(vector::vector::initialized);
    alpha.check(vector::vector::initialized);
    theta.check(vector::vector::allocated);
    phi.check(vector::vector::allocated);
    psi.check(vector::vector::allocated);
    eta.check(vector::vector::allocated);

    if (alpha.if_zero)
        blas::forced_set_const<T, NV>(alpha, 0.0);

    if (r.if_zero)
        blas::set_const<T, NV>(r, 0.0, true);
    if (v.if_zero)
        blas::forced_set_const<T, NV>(v, 0.0);
    if (t.if_zero)
        blas::forced_set_const<T, NV>(t, 0.0);
    if (r0.if_zero)
        blas::forced_set_const<T, NV>(r0, 0.0);

    T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT v_ptr = v.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT t_ptr = t.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT r0_ptr = r0.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    T *XAMG_RESTRICT theta_ptr = theta.get_aligned_ptr<T>();
    T *XAMG_RESTRICT phi_ptr = phi.get_aligned_ptr<T>();
    T *XAMG_RESTRICT psi_ptr = psi.get_aligned_ptr<T>();
    T *XAMG_RESTRICT eta_ptr = eta.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    r.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        theta_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        phi_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        psi_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        eta_ptr[nv] = 0.0;

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            r_ptr[i + nv] -= alpha_ptr[nv] * v_ptr[i + nv];

            theta_ptr[nv] += t_ptr[i + nv] * r_ptr[i + nv];
            phi_ptr[nv] += t_ptr[i + nv] * t_ptr[i + nv];
            psi_ptr[nv] += t_ptr[i + nv] * r0_ptr[i + nv];
            eta_ptr[nv] += r_ptr[i + nv] * r_ptr[i + nv];
        }
    }

    theta.if_initialized = true;
    phi.if_initialized = true;
    psi.if_initialized = true;
    eta.if_initialized = true;

    r.if_zero = false;
    theta.if_zero = false;
    phi.if_zero = false;
    psi.if_zero = false;
    eta.if_zero = false;

    perf.mem_read(4);
    perf.mem_write(1);
    perf.flop(10);
    XAMG_PERF_PRINT_DEBUG_INFO
}

template <typename T, const uint16_t NV>
inline void merged_rbicgstab_group2(vector::vector &x, vector::vector &v_, const vector::vector &t_,
                                    vector::vector &z, const vector::vector &s,
                                    const vector::vector &alpha_conv,
                                    const vector::vector &omega_conv, const vector::vector &alpha,
                                    const vector::vector &omega, const vector::vector &beta) {
    //    23. x = x + \alpha v_  + \omega t_
    //    24. z = t_ - \omega z
    //    25. v_ = z + beta v_ - beta*omega s
    x.check(vector::vector::initialized);
    v_.check(vector::vector::initialized);
    t_.check(vector::vector::initialized);
    z.check(vector::vector::initialized);
    s.check(vector::vector::initialized);
    alpha_conv.check(vector::vector::initialized);
    omega_conv.check(vector::vector::initialized);
    alpha.check(vector::vector::initialized);
    omega.check(vector::vector::initialized);
    beta.check(vector::vector::initialized);

    if (alpha.if_zero)
        blas::forced_set_const<T, NV>(alpha, 0.0);
    if (beta.if_zero)
        blas::forced_set_const<T, NV>(beta, 0.0);
    if (omega.if_zero)
        blas::forced_set_const<T, NV>(omega, 0.0);

    if (x.if_zero)
        blas::set_const<T, NV>(x, 0.0, true);
    if (v_.if_zero)
        blas::set_const<T, NV>(v_, 0.0, true);
    if (t_.if_zero)
        blas::forced_set_const<T, NV>(t_, 0.0);
    if (z.if_zero)
        blas::set_const<T, NV>(z, 0.0, true);
    if (s.if_zero)
        blas::forced_set_const<T, NV>(s, 0.0);

    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    T *XAMG_RESTRICT v__ptr = v_.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT t__ptr = t_.get_aligned_ptr<T>();
    T *XAMG_RESTRICT z_ptr = z.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_conv_ptr = alpha_conv.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_conv_ptr = omega_conv.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_ptr = omega.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT beta_ptr = beta.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    x.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            x_ptr[i + nv] +=
                alpha_conv_ptr[nv] * v__ptr[i + nv] + omega_conv_ptr[nv] * t__ptr[i + nv];
            z_ptr[i + nv] = t__ptr[i + nv] - omega_ptr[nv] * z_ptr[i + nv];
            v__ptr[i + nv] = z_ptr[i + nv] + beta_ptr[nv] * v__ptr[i + nv] -
                             beta_ptr[nv] * omega_ptr[nv] * s_ptr[i + nv];
        }
    }

    x.if_zero = false;
    z.if_zero = false;
    v_.if_zero = false;

    perf.mem_read(5);
    perf.mem_write(3);
    perf.flop(10);
    XAMG_PERF_PRINT_DEBUG_INFO
}

} // namespace specific
} // namespace blas
} // namespace XAMG
