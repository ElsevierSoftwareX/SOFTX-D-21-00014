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
namespace matrix {

template <typename F, typename I1, typename I2, typename I3, typename I4>
std::shared_ptr<backend> create_matrix_block_obj(const csr_matrix<F, I1, I2, I3, I4> &block, const uint16_t &hash, mem::sharing sharing_mode) {
    std::shared_ptr<backend> obj;
    uint16_t F_type, I1_type, I2_type, I3_type, I4_type;
    XAMG::matrix::parse_bit_hash(hash, F_type, I1_type, I2_type, I3_type, I4_type);

#ifdef XAMG_USE_BASIC_INT_TYPES
    I1_type = I32_TYPE;
    I2_type = I32_TYPE;
    I3_type = I32_TYPE;
    I4_type = I32_TYPE;
#endif

#include "csr_matrices_creator.inl"

    return obj;
}


template <typename F, typename I>
std::shared_ptr<backend> create_matrix_block_obj(const ell_matrix<F, I> &block, const uint16_t &hash, mem::sharing sharing_mode) {
    std::shared_ptr<backend> obj;
    uint16_t F_type, I_type;
    XAMG::matrix::parse_bit_hash(hash, F_type, I_type);

    I_type = I32_TYPE;
    F_type = F64_TYPE;

    auto ptr = new ell_matrix<float64_t, uint32_t>(sharing_mode);
    ptr->fill_data(block);
    obj = std::shared_ptr<ell_matrix<float64_t, uint32_t>>(ptr);
#ifdef XAMG_DBG_HEADER
    XAMG::out << XAMG::ALLRANKS << ">> ell_matrix: float64_t, uint32_t" << std::endl;
#endif

    return obj;
}


template <typename F>
std::shared_ptr<backend> create_matrix_block_obj(const dense_matrix<F> &block, const uint16_t &hash, mem::sharing sharing_mode) {
    std::shared_ptr<backend> obj;
    uint16_t F_type;
    XAMG::matrix::parse_bit_hash(hash, F_type);

#include "dense_matrices_creator.inl"

    return obj;
}

}
}
