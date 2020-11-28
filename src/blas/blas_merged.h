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

#include "xamg_headers.h"
#include "xamg_types.h"

namespace XAMG {
namespace blas {
namespace specific {

template <typename T, const uint16_t NV>
inline void merged_bicgstab_group1(vector::vector &r0, const vector::vector &b, vector::vector &p,
                                   vector::vector &r, vector::vector &rho0);

template <typename T, const uint16_t NV>
inline void merged_bicgstab_group2(const vector::vector &t, const vector::vector &s,
                                   vector::vector &phi, vector::vector &psi, vector::vector &theta);

template <typename T, const uint16_t NV>
inline void merged_bicgstab_group4(const vector::vector &s, const vector::vector &t,
                                   const vector::vector &r0, vector::vector &r,
                                   const vector::vector &omega, vector::vector &rho);

template <typename T, const uint16_t NV>
inline void merged_bicgstab_group5(vector::vector &x, vector::vector &p, const vector::vector &s,
                                   const vector::vector &r, const vector::vector &v,
                                   const vector::vector &alpha, const vector::vector &omega,
                                   const vector::vector &beta);

template <typename T, const uint16_t NV>
inline void merged_rbicgstab_group1(vector::vector &r, const vector::vector &v,
                                    const vector::vector &t, const vector::vector &r0,
                                    const vector::vector &alpha, vector::vector &theta,
                                    vector::vector &phi, vector::vector &psi, vector::vector &eta);

template <typename T, const uint16_t NV>
inline void merged_rbicgstab_group2(vector::vector &x, vector::vector &v_, const vector::vector &t_,
                                    vector::vector &z, const vector::vector &s,
                                    const vector::vector &alpha_conv,
                                    const vector::vector &omega_conv, const vector::vector &alpha,
                                    const vector::vector &omega, const vector::vector &beta);

template <typename T, const uint16_t NV>
inline void merged_jacobi(vector::vector &x, const vector::vector &b,
                          const vector::vector &inv_diag, const vector::vector &r,
                          const vector::vector &relax_factor);

//////////

#ifdef XAMG_EXPERIMENTAL_SOLVERS
template <typename T, const uint16_t NV>
inline void merged_pipebicgstab_group1(
    vector::vector &p, const vector::vector &r, vector::vector &s, const vector::vector &w,
    vector::vector &z, const vector::vector &t, const vector::vector &v, vector::vector &q,
    vector::vector &y, const vector::vector &beta, const vector::vector &omega,
    const vector::vector &alpha, vector::vector &theta, vector::vector &phi, vector::vector &pi);

inline void merged_pipebicgstab_group1a(
    vector::vector &p, const vector::vector &r, vector::vector &s, const vector::vector &w,
    vector::vector &z, const vector::vector &t, const vector::vector &v, vector::vector &q,
    vector::vector &y, const vector::vector &beta, const vector::vector &omega,
    const vector::vector &alpha, vector::vector &theta, vector::vector &phi, vector::vector &pi);

template <typename T, const uint16_t NV>
inline void merged_pipebicgstab_group2(vector::vector &x, const vector::vector &p,
                                       const vector::vector &q, vector::vector &r,
                                       const vector::vector &y, const vector::vector &alpha_conv,
                                       const vector::vector &omega_conv,
                                       const vector::vector &alpha, const vector::vector &omega);

template <typename T, const uint16_t NV>
inline void merged_pipebicgstab_group3(vector::vector &w, const vector::vector &y,
                                       const vector::vector &t, const vector::vector &v,
                                       const vector::vector &r0, const vector::vector &r,
                                       const vector::vector &s, const vector::vector &z,
                                       const vector::vector &omega, const vector::vector &alpha,
                                       vector::vector &rho, vector::vector &sigma,
                                       vector::vector &delta, vector::vector &psi);

//////////

template <typename T, const uint16_t NV>
inline void merged_ibicgstab_group1(const vector::vector &r, vector::vector &z, vector::vector &v,
                                    const vector::vector &u, const vector::vector &q,
                                    vector::vector &s, const vector::vector &alpha,
                                    const vector::vector &alpha_1, const vector::vector &beta,
                                    const vector::vector &delta);

template <typename T, const uint16_t NV>
inline void
merged_ibicgstab_group2(vector::vector &t, const vector::vector &u, const vector::vector &q,
                        const vector::vector &r0, const vector::vector &s, const vector::vector &f0,
                        const vector::vector &alpha, vector::vector &phi, vector::vector &pi,
                        vector::vector &gamma, vector::vector &eta, vector::vector &theta,
                        vector::vector &kappa, vector::vector &nu);

template <typename T, const uint16_t NV>
inline void merged_ibicgstab_group3(vector::vector &r, const vector::vector &s,
                                    const vector::vector &t, vector::vector &x,
                                    const vector::vector &z, const vector::vector &omega_conv,
                                    const vector::vector &omega);

//////////

template <typename T, const uint16_t NV>
inline void merged_ppipebicgstab_group1(const vector::vector &r_, vector::vector &s_,
                                        vector::vector &p_, const vector::vector &w_,
                                        const vector::vector &z_, vector::vector &q_,
                                        const vector::vector &alpha, const vector::vector &beta,
                                        const vector::vector &omega);

template <typename T, const uint16_t NV>
inline void merged_ppipebicgstab_group2(const vector::vector &w, vector::vector &s,
                                        vector::vector &z, const vector::vector &t,
                                        const vector::vector &v, const vector::vector &r,
                                        vector::vector &q, vector::vector &y,
                                        const vector::vector &alpha, const vector::vector &beta,
                                        const vector::vector &omega, vector::vector &theta,
                                        vector::vector &phi, vector::vector &pi);

template <typename T, const uint16_t NV>
inline void
merged_ppipebicgstab_group3(vector::vector &x, const vector::vector &p_, const vector::vector &q_,
                            vector::vector &r_, const vector::vector &w_, const vector::vector &z_,
                            const vector::vector &alpha_conv, const vector::vector &omega_conv,
                            const vector::vector &alpha, const vector::vector &omega);

template <typename T, const uint16_t NV>
inline void
merged_ppipebicgstab_group4(vector::vector &r, const vector::vector &q, const vector::vector &y,
                            vector::vector &w, const vector::vector &t, const vector::vector &v,
                            const vector::vector &r0, const vector::vector &s,
                            const vector::vector &z, const vector::vector &alpha,
                            const vector::vector &omega, vector::vector &scal1,
                            vector::vector &sigma, vector::vector &delta, vector::vector &psi);
#endif
} // namespace specific
} // namespace blas
} // namespace XAMG

#include "detail/specific/blas_merged_bicgstab.inl"
#include "detail/specific/blas_merged_rbicgstab.inl"
#ifdef XAMG_EXPERIMENTAL_SOLVERS
#include "detail/specific/blas_merged_ibicgstab.inl"
#include "detail/specific/blas_merged_pipebicgstab.inl"
#include "detail/specific/blas_merged_ppipebicgstab.inl"
#endif
#include "detail/specific/blas_merged_jacobi.inl"
