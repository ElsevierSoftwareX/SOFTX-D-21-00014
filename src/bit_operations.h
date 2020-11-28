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

template <typename T>
uint16_t bit_encoding();

#ifndef XAMG_SEPARATE_OBJECT
// template<> inline uint16_t bit_encoding<float16_t>() { return F16_TYPE; }
template <>
uint16_t bit_encoding<float32_t>() {
    return F32_TYPE;
}
template <>
uint16_t bit_encoding<float64_t>() {
    return F64_TYPE;
}
// template<> inline uint16_t bit_encoding<float128_t>() { return F128_TYPE; }

template <>
uint16_t bit_encoding<uint8_t>() {
    return I8_TYPE;
}
template <>
uint16_t bit_encoding<uint16_t>() {
    return I16_TYPE;
}
template <>
uint16_t bit_encoding<uint32_t>() {
    return I32_TYPE;
}
template <>
uint16_t bit_encoding<uint64_t>() {
    return I64_TYPE;
}
#endif

namespace matrix {

static inline uint16_t create_bit_hash(const uint16_t F) {
    return (F);
}

static inline uint16_t create_bit_hash(const uint16_t F, const uint16_t I1) {
    return (F | (I1 << I1_OFFSET));
}

static inline uint16_t create_bit_hash(const uint16_t F, const uint16_t I1, const uint16_t I2) {
    return (F | (I1 << I1_OFFSET) | (I2 << I2_OFFSET));
}

static inline uint16_t create_bit_hash(const uint16_t F, const uint16_t I1, const uint16_t I2,
                                       const uint16_t I3) {
    return (F | (I1 << I1_OFFSET) | (I2 << I2_OFFSET) | (I3 << I3_OFFSET));
}

static inline uint16_t create_bit_hash(const uint16_t F, const uint16_t I1, const uint16_t I2,
                                       const uint16_t I3, const uint16_t I4) {
    return (F | (I1 << I1_OFFSET) | (I2 << I2_OFFSET) | (I3 << I3_OFFSET) | (I4 << I4_OFFSET));
}

static inline void parse_bit_hash(const uint16_t hash, uint16_t &F) {
    F = hash & F_MASK;
}

static inline void parse_bit_hash(const uint16_t hash, uint16_t &F, uint16_t &I1) {
    //    F = hash | F_MASK;
    parse_bit_hash(hash, F);
    I1 = (hash & I1_MASK) >> I1_OFFSET;
}

static inline void parse_bit_hash(const uint16_t hash, uint16_t &F, uint16_t &I1, uint16_t &I2) {
    parse_bit_hash(hash, F, I1);
    I2 = (hash & I2_MASK) >> I2_OFFSET;
}

static inline void parse_bit_hash(const uint16_t hash, uint16_t &F, uint16_t &I1, uint16_t &I2,
                                  uint16_t &I3) {
    parse_bit_hash(hash, F, I1, I2);
    I3 = (hash & I3_MASK) >> I3_OFFSET;
}

static inline void parse_bit_hash(const uint16_t hash, uint16_t &F, uint16_t &I1, uint16_t &I2,
                                  uint16_t &I3, uint16_t &I4) {
    parse_bit_hash(hash, F, I1, I2, I3);
    I4 = (hash & I4_MASK) >> I4_OFFSET;
}

}; // namespace matrix

static inline uint16_t define_int_type(const uint64_t &data_range) {

    if (data_range <= UINT8_MAX)
        return I8_TYPE;
    else if (data_range <= UINT16_MAX)
        return I16_TYPE;
    else if (data_range <= UINT32_MAX)
        return I32_TYPE;
    else
        return I64_TYPE;
}

// static inline uint16_t encode_csr_hash(const uint64_t nonzeros, uint64_t ncols, uint64_t block_nrows, uint64_t block_ncols) {
//
//    uint16_t hash = (define_int_type(nonzeros) << I1_OFFSET) |
//                    (define_int_type(ncols) << I2_OFFSET) |
//                    (define_int_type(block_nrows) << I3_OFFSET) |
//                    (define_int_type(block_ncols) << I4_OFFSET);
//
//    return hash;
//}

// static inline uint16_t encode_ell_hash(...) {
//
//    ...
//
//    return hash;
//}

} // namespace XAMG
