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
namespace blas2 {

namespace backend {
template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
void csr_Axpy(const matrix::csr_matrix<F, I1, I2, I3, I4> &m, const vector::vector &x,
              vector::vector &y, uint16_t nv);

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
void csr_Ax_y(const matrix::csr_matrix<F, I1, I2, I3, I4> &m, const vector::vector &x,
              vector::vector &y, uint16_t nv);

template <typename F, uint16_t NV>
void dense_Axpy(const matrix::dense_matrix<F> &m, const vector::vector &x, vector::vector &y,
                uint16_t nv);

template <typename F, uint16_t NV>
void dense_Ax_y(const matrix::dense_matrix<F> &m, const vector::vector &x, vector::vector &y,
                uint16_t nv);

template <typename F, typename I, uint16_t NV>
void ell_Axpy(const matrix::ell_matrix<F, I> &m, const vector::vector &x, vector::vector &y,
              uint16_t nv);

template <typename F, typename I, uint16_t NV>
void ell_Ax_y(const matrix::ell_matrix<F, I> &m, const vector::vector &x, vector::vector &y,
              uint16_t nv);

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
void csr_SGS(const matrix::csr_matrix<F, I1, I2, I3, I4> &m, const vector::vector &inv_diag,
             const vector::vector &t, vector::vector &x, const vector::vector &relax_factor,
             uint16_t nv);

template <typename F, uint16_t NV>
void dense_SGS(const matrix::dense_matrix<F> &m, const vector::vector &inv_diag,
               const vector::vector &t, vector::vector &x, const vector::vector &relax_factor,
               uint16_t nv);

template <typename F, typename I, uint16_t NV>
void ell_SGS(const matrix::ell_matrix<F, I> &m, const vector::vector &inv_diag,
             const vector::vector &t, vector::vector &x, const vector::vector &relax_factor,
             uint16_t nv);
} // namespace backend

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
struct blas2_matvec : public XAMG::matrix::matrix_op_base {
    virtual void csr_Axpy(void *p, const vector::vector &x, vector::vector &y, uint16_t nv) {
        assert(p != nullptr);
        const matrix::csr_matrix<F, I1, I2, I3, I4> *m = (matrix::csr_matrix<F, I1, I2, I3, I4> *)p;
        backend::csr_Axpy<F, I1, I2, I3, I4, NV>(*m, x, y, nv);
    }
    virtual void csr_Ax_y(void *p, const vector::vector &x, vector::vector &y, uint16_t nv) {
        assert(p != nullptr);
        const matrix::csr_matrix<F, I1, I2, I3, I4> *m = (matrix::csr_matrix<F, I1, I2, I3, I4> *)p;
        backend::csr_Ax_y<F, I1, I2, I3, I4, NV>(*m, x, y, nv);
    }
    virtual void ell_Ax_y(void *p, const vector::vector &x, vector::vector &y, uint16_t nv) {
        assert(p != nullptr);
        const matrix::ell_matrix<F, I1> *m = (matrix::ell_matrix<F, I1> *)p;
        backend::ell_Ax_y<F, I1, NV>(*m, x, y, nv);
    }
    virtual void ell_Axpy(void *p, const vector::vector &x, vector::vector &y, uint16_t nv) {
        assert(p != nullptr);
        const matrix::ell_matrix<F, I1> *m = (matrix::ell_matrix<F, I1> *)p;
        backend::ell_Axpy<F, I1, NV>(*m, x, y, nv);
    }
    virtual void dense_Ax_y(void *p, const vector::vector &x, vector::vector &y, uint16_t nv) {
        assert(p != nullptr);
        const matrix::dense_matrix<F> *m = (matrix::dense_matrix<F> *)p;
        backend::dense_Ax_y<F, NV>(*m, x, y, nv);
    }
    virtual void dense_Axpy(void *p, const vector::vector &x, vector::vector &y, uint16_t nv) {
        assert(p != nullptr);
        const matrix::dense_matrix<F> *m = (matrix::dense_matrix<F> *)p;
        backend::dense_Axpy<F, NV>(*m, x, y, nv);
    }

    virtual void csr_SGS(void *p, const vector::vector &inv_diag, const vector::vector &t,
                         vector::vector &x, const vector::vector &relax_factor, uint16_t nv) {
        assert(p != nullptr);
        const matrix::csr_matrix<F, I1, I2, I3, I4> *m = (matrix::csr_matrix<F, I1, I2, I3, I4> *)p;
        backend::csr_SGS<F, I1, I2, I3, I4, NV>(*m, inv_diag, t, x, relax_factor, nv);
    }
    virtual void dense_SGS(void *p, const vector::vector &inv_diag, const vector::vector &t,
                           vector::vector &x, const vector::vector &relax_factor, uint16_t nv) {
        assert(p != nullptr);
        const matrix::dense_matrix<F> *m = (matrix::dense_matrix<F> *)p;
        backend::dense_SGS<F, NV>(*m, inv_diag, t, x, relax_factor, nv);
    }
    virtual void ell_SGS(void *p, const vector::vector &inv_diag, const vector::vector &t,
                         vector::vector &x, const vector::vector &relax_factor, uint16_t nv) {
        assert(p != nullptr);
        const matrix::ell_matrix<F, I1> *m = (matrix::ell_matrix<F, I1> *)p;
        backend::ell_SGS<F, I1, NV>(*m, inv_diag, t, x, relax_factor, nv);
    }
};

} // namespace blas2
} // namespace XAMG

#include "detail/csr_matvec.inl"
#include "detail/dense_matvec.inl"
#include "detail/ell_matvec.inl"

#include "detail/csr_gs.inl"
#include "detail/dense_gs.inl"
#include "detail/ell_gs.inl"
