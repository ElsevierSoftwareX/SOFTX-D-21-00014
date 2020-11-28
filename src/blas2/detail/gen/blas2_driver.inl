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

template <const uint16_t NV>
std::shared_ptr<matrix::matrix_op_base> create_blas2_driver_obj(uint16_t &hash) {

    std::shared_ptr<matrix::matrix_op_base> obj;
    uint16_t F_type, I1_type, I2_type, I3_type, I4_type;
    XAMG::matrix::parse_bit_hash(hash, F_type, I1_type, I2_type, I3_type, I4_type);

/////////
#ifdef XAMG_USE_BASIC_INT_TYPES
    I1_type = I32_TYPE;
    I2_type = I32_TYPE;
    I3_type = I32_TYPE;
    I4_type = I32_TYPE;
#endif

#include "blas2_drivers_creator.inl"

    return obj;
}

template <const uint16_t NV>
void generate_blas2_layer_drivers(matrix::segmentation_layer &layer, bool diag_flag) {
    for (uint32_t nb = 0; nb < layer.offd.size(); nb++)
        layer.offd[nb].blas2_driver = create_blas2_driver_obj<NV>(layer.offd[nb].hash);

    if (diag_flag)
        layer.diag.blas2_driver = create_blas2_driver_obj<NV>(layer.diag.hash);
}

template <const uint16_t NV>
void generate_blas2_drivers(matrix::matrix &m) {
//    std::cout << FUNC_PREFIX << "function \n";
    if (m.data_layer.find(segment::NODE) != m.data_layer.end())
        generate_blas2_layer_drivers<NV>(m.data_layer.find(segment::NODE)->second, false);
    if (m.data_layer.find(segment::NUMA) != m.data_layer.end())
        generate_blas2_layer_drivers<NV>(m.data_layer.find(segment::NUMA)->second, true);
    if (m.data_layer.find(segment::CORE) != m.data_layer.end())
        generate_blas2_layer_drivers<NV>(m.data_layer.find(segment::CORE)->second, true);

    for (auto &layer : m.data_layer) {
        layer.second.p2p_comm.send.alloc_tokens();
        layer.second.p2p_comm.recv.alloc_tokens();
    }

    m.if_drivers_allocated = true;
}

}
}
