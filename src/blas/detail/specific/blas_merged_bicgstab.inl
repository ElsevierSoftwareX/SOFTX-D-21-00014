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
inline void merged_bicgstab_group1(vector::vector &r0, const vector::vector &b, vector::vector &p,
                                   vector::vector &r, vector::vector &rho0) {
    //    2. r0 = b - r0; \rho_0 = (r0, r0); p = r0; r = r0
    r0.check(vector::vector::initialized);
    b.check(vector::vector::initialized);
    p.check(vector::vector::allocated);
    r.check(vector::vector::allocated);
    rho0.check(vector::vector::allocated);

    if (b.if_zero)
        //        assert(0 && "unexpected behaviour: RHS vector is zero!");
        blas::forced_set_const<T, NV>(b, 0.0);
    if (r0.if_zero)
        blas::set_const<T, NV>(r0, 0.0, true);

    T *XAMG_RESTRICT r0_ptr = r0.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT b_ptr = b.get_aligned_ptr<T>();
    T *XAMG_RESTRICT p_ptr = p.get_aligned_ptr<T>();
    T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();

    T *XAMG_RESTRICT rho0_ptr = rho0.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    r0.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        rho0_ptr[nv] = 0.0;

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            r0_ptr[i + nv] = b_ptr[i + nv] - r0_ptr[i + nv];
            rho0_ptr[nv] += r0_ptr[i + nv] * r0_ptr[i + nv];
            p_ptr[i + nv] = r0_ptr[i + nv];
            r_ptr[i + nv] = r0_ptr[i + nv];
        }
    }

    rho0.if_initialized = true;
    p.if_initialized = true;
    r.if_initialized = true;

    r0.if_zero = false;
    rho0.if_zero = false;
    p.if_zero = false;
    r.if_zero = false;

    perf.mem_read(2);
    perf.mem_write(3);
    perf.flop(6);
    XAMG_PERF_PRINT_DEBUG_INFO
}

template <typename T, const uint16_t NV>
inline void merged_bicgstab_group2(const vector::vector &t, const vector::vector &s,
                                   vector::vector &phi, vector::vector &psi,
                                   vector::vector &theta) {
    //    8. \phi = (t, s); \psi = (t, t); \theta = (s, s)
    t.check(vector::vector::initialized);
    s.check(vector::vector::initialized);
    phi.check(vector::vector::allocated);
    psi.check(vector::vector::allocated);
    theta.check(vector::vector::allocated);

    if (t.if_zero)
        blas::forced_set_const<T, NV>(t, 0.0);
    if (s.if_zero)
        blas::forced_set_const<T, NV>(s, 0.0);

    const T *XAMG_RESTRICT t_ptr = t.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();

    T *XAMG_RESTRICT phi_ptr = phi.get_aligned_ptr<T>();
    T *XAMG_RESTRICT psi_ptr = psi.get_aligned_ptr<T>();
    T *XAMG_RESTRICT theta_ptr = theta.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    s.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        phi_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        psi_ptr[nv] = 0.0;
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        theta_ptr[nv] = 0.0;

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            phi_ptr[nv] += t_ptr[i + nv] * s_ptr[i + nv];
            psi_ptr[nv] += t_ptr[i + nv] * t_ptr[i + nv];
            theta_ptr[nv] += s_ptr[i + nv] * s_ptr[i + nv];
        }
    }

    phi.if_initialized = true;
    psi.if_initialized = true;
    theta.if_initialized = true;

    phi.if_zero = false;
    psi.if_zero = false;
    theta.if_zero = false;

    perf.mem_read(2);
    perf.flop(6);
    XAMG_PERF_PRINT_DEBUG_INFO
}

template <typename T, const uint16_t NV>
inline void merged_bicgstab_group4(const vector::vector &s, const vector::vector &t,
                                   const vector::vector &r0, vector::vector &r,
                                   const vector::vector &omega, vector::vector &rho) {
    //    14. r = s - \omega t; \rho_n = (r, r0)

    if (s.if_zero)
        blas::forced_set_const<T, NV>(s, 0.0);
    if (t.if_zero)
        blas::forced_set_const<T, NV>(t, 0.0);
    if (r0.if_zero)
        blas::forced_set_const<T, NV>(r0, 0.0);

    const T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT t_ptr = t.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT r0_ptr = r0.get_aligned_ptr<T>();
    T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT omega_ptr = omega.get_aligned_ptr<T>();
    T *XAMG_RESTRICT rho_ptr = rho.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    s.get_core_range<T>(core_size, core_offset);

    //////////

    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv)
        rho_ptr[nv] = 0.0;

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            r_ptr[i + nv] = s_ptr[i + nv] - omega_ptr[nv] * t_ptr[i + nv];
            rho_ptr[nv] += r_ptr[i + nv] * r0_ptr[i + nv];
        }
    }

    r.if_zero = false;
    rho.if_zero = false;

    perf.mem_read(3);
    perf.mem_write(1);
    perf.flop(4);
    XAMG_PERF_PRINT_DEBUG_INFO
}

//////////

template <typename T, const uint16_t NV>
inline void merged_bicgstab_group5(vector::vector &x, vector::vector &p, const vector::vector &s,
                                   const vector::vector &r, const vector::vector &v,
                                   const vector::vector &alpha_conv,
                                   const vector::vector &omega_conv, const vector::vector &alpha,
                                   const vector::vector &omega, const vector::vector &beta) {
    //    16. x = x + \alpha p + \omega s; p = r + \beta p - \beta*\omega v

    if (x.if_zero)
        blas::set_const<T, NV>(x, 0.0, true);
    if (p.if_zero)
        blas::set_const<T, NV>(p, 0.0, true);
    if (s.if_zero)
        blas::forced_set_const<T, NV>(s, 0.0);
    if (r.if_zero)
        blas::forced_set_const<T, NV>(r, 0.0);
    if (v.if_zero)
        blas::forced_set_const<T, NV>(v, 0.0);

    T *XAMG_RESTRICT x_ptr = x.get_aligned_ptr<T>();
    T *XAMG_RESTRICT p_ptr = p.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT s_ptr = s.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT r_ptr = r.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT v_ptr = v.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_ptr = alpha.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_ptr = omega.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT beta_ptr = beta.get_aligned_ptr<T>();

    const T *XAMG_RESTRICT alpha_conv_ptr = alpha_conv.get_aligned_ptr<T>();
    const T *XAMG_RESTRICT omega_conv_ptr = omega_conv.get_aligned_ptr<T>();

    uint64_t core_size, core_offset;
    s.get_core_range<T>(core_size, core_offset);

    XAMG_STACK_ALIGNMENT_PREFIX T val[NV];
    XAMG_VECTOR_ALIGN
    for (uint16_t nv = 0; nv < NV; ++nv) {
        val[nv] = beta_ptr[nv] * omega_ptr[nv];
    }

    //////////

    XAMG_VECTOR_ALIGN
    for (uint64_t i = NV * core_offset; i < NV * (core_offset + core_size); i += NV) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            x_ptr[i + nv] +=
                alpha_conv_ptr[nv] * p_ptr[i + nv] + omega_conv_ptr[nv] * s_ptr[i + nv];
            p_ptr[i + nv] = r_ptr[i + nv] + beta_ptr[nv] * p_ptr[i + nv] - val[nv] * v_ptr[i + nv];
        }
    }

    x.if_zero = false;
    p.if_zero = false;

    perf.mem_read(5);
    perf.mem_write(2);
    perf.flop(8);
    XAMG_PERF_PRINT_DEBUG_INFO
}

} // namespace specific
} // namespace blas
} // namespace XAMG
