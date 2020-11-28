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

#include "cube.inl"
#include "channel_with_cube.inl"

struct generator_params_t {
    std::string test_case = "cube";
    uint16_t nx = 10, ny = 10, nz = 1; // valid for CUBE case
    uint8_t scale = 1;                 // valid for CHANNEL_WITH_CUBE case
    bool const_rhs = 1;                // valid for CHANNEL_WITH_CUBE case
};

static inline bool parse_generator_params(const std::map<std::string, std::string> &kvmap,
                                          generator_params_t &p) {
    for (auto kv : kvmap) {
        auto key = kv.first;
        auto value = kv.second;
        if (key == "case") {
            p.test_case = value;
        } else {
            std::regex unsigned_integer_number("^[0-9]*$", std::regex_constants::ECMAScript);
            if (std::regex_search(value, unsigned_integer_number)) {
                uint16_t n = std::stoi(value);
                if (key == "nx") {
                    p.nx = n;
                } else if (key == "ny") {
                    p.ny = n;
                } else if (key == "nz") {
                    p.nz = n;
                } else if (key == "scale") {
                    p.scale = n;
                } else if (key == "const_rhs") {
                    p.const_rhs = n;
                } else {
                    return false;
                }
            }
        }
    }
    assert((p.test_case == "cube") || (p.test_case == "channel_with_cube"));
    return true;
}

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
bool generate_system(XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> &mat, XAMG::vector::vector &x,
                     XAMG::vector::vector &b, const generator_params_t &generator_params) {
    assert(x.sharing_mode == XAMG::mem::CORE);
    assert(b.sharing_mode == XAMG::mem::CORE);

    if (generator_params.test_case == "cube") {
        generate_cube<F, I1, I2, I3, I4, NV>(mat, x, b, generator_params.nx, generator_params.ny,
                                             generator_params.nz);
    } else if (generator_params.test_case == "channel_with_cube") {
        generate_channel_with_cube<F, I1, I2, I3, I4, NV>(mat, x, b, generator_params.scale,
                                                          generator_params.const_rhs);
    }

    //    x.alloc<F>(mat.nrows, NV);
    //    b.alloc<F>(mat.nrows, NV);
    //    x.ext_offset = b.ext_offset = block_offset;
    //
    //    blas::set_const<F, NV>(x, 0.0, true);
    ////    blas::set_rand<F, NV>(x, false);
    //    blas::set_const<F, NV>(b, 1.0, true);
    ////    blas::set_rand<F, NV>(b, false);
    //    XAMG::io::sync();

    return true;
}

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
bool generate_system(XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> &mat, XAMG::vector::vector &x,
                     XAMG::vector::vector &b, const size_t nx, const size_t ny, const size_t nz) {
    generator_params_t generator_params;
    generator_params.nx = generator_params.ny = generator_params.nz = 10;

    return (generate_system<F, I1, I2, I3, I4, NV>(mat, x, b, generator_params));
}
