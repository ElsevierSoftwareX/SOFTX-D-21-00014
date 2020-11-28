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

#include "lame.inl"

template <typename F, uint16_t NV>
std::vector<F> generate_velocity(const uint64_t indx) {
    std::vector<F> vel;
    for (uint16_t nv = 0; nv < NV; ++nv)
        vel.push_back(0.1 * XAMG::pseudo_rand<F>(indx * NV + nv));

    return vel;
}

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
bool generate_channel_with_cube(XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> &mat,
                                XAMG::vector::vector &x, XAMG::vector::vector &b,
                                const uint8_t grid_scale, const bool const_rhs) {
    cube_geometry geom(grid_scale);
    const uint32_t nx = geom.grid.nx;
    const uint32_t ny = geom.grid.ny;
    const uint32_t nz = geom.grid.nz;

    uint32_t i32_nrows, i32_nonzeros;
    i32_nrows = nx * ny * nz;

    uint64_t block_size = i32_nrows / id.gl_nprocs;
    uint64_t block_offset = block_size * id.gl_proc;
    if (id.gl_proc == id.gl_nprocs - 1)
        block_size = i32_nrows - block_offset;

    //////////
    //    getting block nonzeros:

    uint64_t block_nonzeros = 0;
    for (uint32_t indx = block_offset; indx < block_offset + block_size; ++indx) {
        uint64_t ii = indx % nx;
        uint64_t jj = (indx / nx) % ny;
        uint64_t kk = indx / (nx * ny);

        ++block_nonzeros; // diag

        if (!geom.grid.cube.is_solid(ii, jj, kk)) {
            if (!geom.grid.cube.is_solid(ii, jj, kk - 1))
                ++block_nonzeros;
            if (!geom.grid.cube.is_solid(ii, jj - 1, kk) && !geom.is_ext_ywall(jj - 1))
                ++block_nonzeros;
            if (!geom.grid.cube.is_solid(ii - 1, jj, kk))
                ++block_nonzeros;

            if (!geom.grid.cube.is_solid(ii + 1, jj, kk))
                ++block_nonzeros;
            if (!geom.grid.cube.is_solid(ii, jj + 1, kk) && !geom.is_ext_ywall(jj + 1))
                ++block_nonzeros;
            if (!geom.grid.cube.is_solid(ii, jj, kk + 1))
                ++block_nonzeros;
        }
    }

    //////////

    mat.nrows = block_size;
    mat.block_nrows = block_size;
    mat.ncols = i32_nrows;
    mat.block_ncols = i32_nrows;
    mat.block_row_offset = block_offset;
    mat.block_col_offset = 0;
    mat.nonzeros = block_nonzeros;

    mat.alloc();
    // XAMG::out << XAMG::ALLRANKS << XAMG::DBG << "Local: " << mat.nrows << " || " << mat.nonzeros
    //           << std::endl;

    /////////

    uint64_t row_offset = mat.block_row_offset;

    uint64_t cntr = 0;
    for (uint32_t indx = block_offset; indx < block_offset + block_size; ++indx) {
        uint64_t ii = indx % nx;
        uint64_t jj = (indx / nx) % ny;
        uint64_t kk = indx / (nx * ny);

        std::vector<F> vals;
        std::vector<uint64_t> inds;

        vals.push_back(0.0);
        inds.push_back(indx);

        if (!geom.grid.cube.is_solid(ii, jj, kk)) {
            F q_e = (geom.H_f1(ii + 1, jj, kk) / std::pow(geom.H1_f1(ii + 1), 2.)) /
                    (geom.hx * geom.hx);
            F q_w = (geom.H_f1(ii, jj, kk) / std::pow(geom.H1_f1(ii), 2.)) / (geom.hx * geom.hx);
            F q_n = (geom.H_f2(ii, jj + 1, kk) / std::pow(geom.H2_f2(jj + 1), 2.)) /
                    (geom.hy * geom.hy);
            F q_s = (geom.H_f2(ii, jj, kk) / std::pow(geom.H2_f2(jj), 2.)) / (geom.hy * geom.hy);
            F q_t = (geom.H_f3(ii, jj, kk + 1) / std::pow(geom.H3_f3(kk + 1), 2.)) /
                    (geom.hz * geom.hz);
            F q_b = (geom.H_f3(ii, jj, kk) / std::pow(geom.H3_f3(kk), 2.)) / (geom.hz * geom.hz);

            if (!geom.grid.cube.is_solid(ii, jj, kk - 1)) {
                vals[0] += q_b;
                vals.push_back(-q_b);
                inds.push_back(((kk - 1 + nz) % nz) * nx * ny + jj * nx + ii);
            }
            if (!geom.grid.cube.is_solid(ii, jj - 1, kk) && !geom.is_ext_ywall(jj - 1)) {
                vals[0] += q_s;
                vals.push_back(-q_s);
                inds.push_back(kk * nx * ny + (jj - 1) * nx + ii);
            }
            if (!geom.grid.cube.is_solid(ii - 1, jj, kk)) {
                vals[0] += q_w;
                vals.push_back(-q_w);
                inds.push_back(kk * nx * ny + jj * nx + (ii - 1 + nx) % nx);
            }

            if (!geom.grid.cube.is_solid(ii + 1, jj, kk)) {
                vals[0] += q_e;
                vals.push_back(-q_e);
                inds.push_back(kk * nx * ny + jj * nx + (ii + 1) % nx);
            }
            if (!geom.grid.cube.is_solid(ii, jj + 1, kk) && !geom.is_ext_ywall(jj + 1)) {
                vals[0] += q_n;
                vals.push_back(-q_n);
                inds.push_back(kk * nx * ny + (jj + 1) * nx + ii);
            }
            if (!geom.grid.cube.is_solid(ii, jj, kk + 1)) {
                vals[0] += q_t;
                vals.push_back(-q_t);
                inds.push_back(((kk + 1) % nz) * nx * ny + jj * nx + ii);
            }
        } else {
            vals[0] = 1.0;
        }

        if (indx == i32_nrows - 1) {
            vals[0] = 1.0;
            for (size_t i = 1; i < vals.size(); ++i)
                vals[i] = 0.0;
        }

        double scal1 = 1. / geom.H_c(ii, jj, kk);
        for (auto &val : vals)
            val *= scal1;

        double scal2 = geom.H_c(ii, jj, kk) * geom.hx * geom.hy * geom.hz;
        for (auto &val : vals)
            val *= scal2;

        mat.upload_row(indx - block_offset, inds, vals, vals.size());
        cntr += vals.size();
    }

    assert(block_nonzeros == cntr);
    // std::cout << "nonzeros filled: " << cntr << "\n";
    // mat.nonzeros = cntr;
    // mat.realloc(); // we can shrink vectors here

    //////////

    x.alloc<F>(mat.nrows, NV);
    b.alloc<F>(mat.nrows, NV);
    x.ext_offset = b.ext_offset = block_offset;

    XAMG::blas::set_const<F, NV>(x, 0.0, true);

    /////////

    if (const_rhs) {
        XAMG::blas::set_const<F, NV>(b, 1.0, true);
    } else {
        auto b_ptr = b.get_aligned_ptr<F>();

        double inv_hx = 1. / geom.hx;
        double inv_hy = 1. / geom.hy;
        double inv_hz = 1. / geom.hz;

        for (uint32_t indx = block_offset; indx < block_offset + block_size; ++indx) {
            uint64_t ii = indx % nx;
            uint64_t jj = (indx / nx) % ny;
            uint64_t kk = indx / (nx * ny);

            F H_c = geom.H_c(ii, jj, kk);
            F H23_w = geom.H_f1(ii, jj, kk) / geom.H1_f1(ii);
            F H23_e = geom.H_f1(ii + 1, jj, kk) / geom.H1_f1(ii + 1);
            F H13_s = geom.H_f2(ii, jj, kk) / geom.H2_f2(jj);
            F H13_n = geom.H_f2(ii, jj + 1, kk) / geom.H2_f2(jj + 1);
            F H12_b = geom.H_f3(ii, jj, kk) / geom.H3_f3(kk);
            F H12_t = geom.H_f3(ii, jj, kk + 1) / geom.H3_f3(kk + 1);
            std::vector<F> u_e = generate_velocity<F, NV>(kk * geom.grid.ny * (geom.grid.nx + 1) +
                                                          jj * (geom.grid.nx + 1) + ii + 1);
            std::vector<F> u_w = generate_velocity<F, NV>(kk * geom.grid.ny * (geom.grid.nx + 1) +
                                                          jj * (geom.grid.nx + 1) + ii);
            std::vector<F> v_n = generate_velocity<F, NV>(kk * (geom.grid.ny + 1) * geom.grid.nx +
                                                          (jj + 1) * geom.grid.nx + ii);
            std::vector<F> v_s = generate_velocity<F, NV>(kk * (geom.grid.ny + 1) * geom.grid.nx +
                                                          jj * geom.grid.nx + ii);
            std::vector<F> w_t = generate_velocity<F, NV>((kk + 1) * geom.grid.ny * geom.grid.nx +
                                                          jj * geom.grid.nx + ii);
            std::vector<F> w_b =
                generate_velocity<F, NV>(kk * geom.grid.ny * geom.grid.nx + jj * geom.grid.nx + ii);

            for (size_t nv = 0; nv < NV; ++nv) {
                b_ptr[(indx - block_offset) * NV + nv] =
                    -((H23_e * u_e[nv] - H23_w * u_w[nv]) * inv_hx +
                      (H13_n * v_n[nv] - H13_s * v_s[nv]) * inv_hy +
                      (H12_t * w_t[nv] - H12_b * w_b[nv]) * inv_hz) /
                    H_c;
            }

            if (indx == i32_nrows - 1) {
                for (size_t nv = 0; nv < NV; nv++)
                    b_ptr[(indx - block_offset) * NV + nv] = 0.;
            }
        }
        b.if_initialized = true;
        b.if_zero = false;
    }

    return true;
}
