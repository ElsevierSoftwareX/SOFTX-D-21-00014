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

#include <sstream>

namespace XAMG {

namespace misc {

template <typename T>
uint32_t search_range(const T col, const std::vector<uint64_t> &range) {

    uint32_t val = range.size() - 1;
    //    for (uint64_t i = 0; i < range.size(); ++i) {
    while ((col < range[val]) && (val > 0)) {
        --val;
    }

    return val;
}

template <typename T>
uint64_t split_range(const uint64_t &size, const int &nblocks) {
    uint32_t factor = XAMG_ALIGN_SIZE / sizeof(T);
    assert(factor);

    uint64_t core_size = size / nblocks;
    core_size += factor - (core_size % factor);

    return core_size;
}

static inline uint32_t hash(char *str) {
    uint32_t hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

static inline void str_split(const std::string &s, char delimiter,
                             std::vector<std::string> &result) {
    std::string token;
    std::istringstream token_stream(s);
    while (std::getline(token_stream, token, delimiter)) {
        result.push_back(token);
    }
}

static inline std::vector<std::string> str_split(const std::string &s, char delimiter) {
    std::vector<std::string> result;
    std::string token;
    std::istringstream token_stream(s);
    while (std::getline(token_stream, token, delimiter)) {
        result.push_back(token);
    }
    return result;
}

static inline void vstr_to_vint(std::vector<std::string> &from, std::vector<int> &to) {
    to.clear();
    for (auto &s : from) {
        int x = std::stoi(s);
        to.push_back(x);
    }
}

} // namespace misc
} // namespace XAMG
