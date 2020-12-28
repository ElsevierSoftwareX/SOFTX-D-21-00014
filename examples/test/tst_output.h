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

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include "yaml-cpp/yaml.h"

struct tst_store_output {
    YAML::Emitter out;
    std::map<std::string, YAML::Node> nodes;
    void init() {
        out << YAML::BeginDoc;
        out << YAML::BeginMap;
    }
    template <typename T>
    void store_item(const std::string &block, const std::string &key, const T &value) {
        YAML::Node &n = nodes[block];
        n[key] = value;
        n[key].SetStyle(YAML::EmitterStyle::Flow);
    }
    template <typename T>
    void store_xamg_vector(const std::string &block, const std::string &key,
                           const XAMG::vector::vector &vec, int i = 0) {
        std::vector<T> vec_stdv = vec.get_element<T>(i);
        YAML::Node &n = nodes[block];
        n[key] = vec_stdv;
        n[key].SetStyle(YAML::EmitterStyle::Flow);
    }
    template <typename T, int NV>
    void store_xamg_vector_norm(const std::string &block, const std::string &key,
                                const XAMG::vector::vector &vec) {
        XAMG::vector::vector res(XAMG::mem::LOCAL);
        res.alloc<T>(1, NV);
        XAMG::blas::dot_global<T, NV>(vec, vec, res);
        std::vector<T> res_stdv = vec.get_element<T>(0);
        std::transform(res_stdv.begin(), res_stdv.end(), res_stdv.begin(),
                       [](float64_t x) { return sqrt(x); });
        YAML::Node &n = nodes[block];
        n[key] = res_stdv;
        n[key].SetStyle(YAML::EmitterStyle::Flow);
    }
    void dump(const std::string &output_filename) {
        if (output_filename == "")
            return;
        std::ofstream ofs(output_filename);
        dump(ofs);
        ofs.close();
    }
    void dump(std::ostream &ofs) {
        for (auto &n : nodes) {
            out << YAML::Flow << YAML::Key << n.first << YAML::Value << n.second;
        }
        out << YAML::Newline;
        ofs << std::string(out.c_str());
    }
};
