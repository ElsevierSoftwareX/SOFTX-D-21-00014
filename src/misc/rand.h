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

inline uint32_t xorshift32(uint32_t i) {
    uint32_t state = i + 1;
    //    state = state * 1000 + 7;
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

inline uint64_t xorshift64(uint64_t i) {
    uint64_t state = i + 1;
    //    state = state * 1000 + 7;
    state ^= (state << 13);
    state ^= (state >> 7);
    state ^= (state << 17);
    return state;
}

inline int fastrand(int g_seed) {
    //    g_seed = 214013 * g_seed + 2531011;
    //    return (g_seed >> 16) & 0x7FFF;
    g_seed = 1664525 * g_seed + 1013904223;
    return g_seed;
}

inline int pseudo_rand(uint32_t i) {
    return fastrand(xorshift32(i));
}

template <typename T>
T true_rand() {
    T val = rand();
    return (val / RAND_MAX);
}

template <typename T>
T pseudo_rand(const uint64_t i) {
    //    T val = xorshift64(i);
    T val = pseudo_rand(i);
    return 0.5 * (1.0 + val / INT_MAX);
}

} // namespace XAMG
