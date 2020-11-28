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
namespace vector {

template <typename T1, typename T2>
void convert(const vector &v1, vector &v2) {
#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "Converting type <" << DEMANGLE_TYPEID_NAME(T2) << "> to type <"
              << DEMANGLE_TYPEID_NAME(T1) << ">\n";
#endif
    v2.check(vector::nonempty);

    bool active_proc = false;
    if (v2.sharing_mode == mem::NUMA)
        active_proc = id.numa_master_process();
    else if (v2.sharing_mode == mem::NODE)
        active_proc = id.node_master_process();
    else
        active_proc = true;

    if (active_proc)
        v1.check(vector::initialized);

    v2.if_zero = v1.if_zero;
    if (v1.if_zero) {
        v2.if_initialized = true;
        return;
    }

    const T1 *XAMG_RESTRICT v1_ptr = v1.get_aligned_ptr<T1>();
    T2 *XAMG_RESTRICT v2_ptr = v2.get_aligned_ptr<T2>();

    assert(((void *)v1_ptr) != ((void *)v2_ptr));
    assert(v2.size == v1.size);
    assert(v2.nv == v1.nv);

    if (active_proc) {
        if ((v1.sharing_mode == mem::NUMA_NODE) && (v2.sharing_mode == mem::NUMA_NODE)) {
            uint64_t core_size, core_offset;
            v1.get_core_range<T1>(core_size, core_offset);

            // XAMG_VECTOR_NODEP
            XAMG_VECTOR_ALIGN
            for (uint64_t i = v2.nv * core_offset; i < v2.nv * (core_offset + core_size); i++) {
                v2_ptr[i] = (T2)v1_ptr[i];
            }
        } else {
            // XAMG_VECTOR_NODEP
            XAMG_VECTOR_ALIGN
            for (uint64_t i = 0; i < v2.nv * v2.size; i++) {
                v2_ptr[i] = (T2)v1_ptr[i];
            }
        }
    }

    v2.if_initialized = true;
}

template <typename T>
void convert_to(const vector &v1, vector &v2) {
    v1.check(vector::allocated);
    v2.check(vector::allocated);

    if (v2.type_hash == typeid(uint8_t).hash_code()) {
        convert<T, uint8_t>(v1, v2);
    } else if (v2.type_hash == typeid(uint16_t).hash_code()) {
        convert<T, uint16_t>(v1, v2);
    } else if (v2.type_hash == typeid(uint32_t).hash_code()) {
        convert<T, uint32_t>(v1, v2);
    } else if (v2.type_hash == typeid(uint64_t).hash_code()) {
        convert<T, uint64_t>(v1, v2);
    } else if (v2.type_hash == typeid(float32_t).hash_code()) {
        convert<T, float32_t>(v1, v2);
    } else if (v2.type_hash == typeid(float64_t).hash_code()) {
        convert<T, float64_t>(v1, v2);
    } else {
        assert(0);
    }
}

template <typename T>
void convert_from(const vector &v1, vector &v2) {
    // v1.check(vector::allocated);
    assert(v1.is_type_set());
    v2.check(vector::allocated);

    if (v1.type_hash == typeid(uint8_t).hash_code()) {
        convert<uint8_t, T>(v1, v2);
    } else if (v1.type_hash == typeid(uint16_t).hash_code()) {
        convert<uint16_t, T>(v1, v2);
    } else if (v1.type_hash == typeid(uint32_t).hash_code()) {
        convert<uint32_t, T>(v1, v2);
    } else if (v1.type_hash == typeid(uint64_t).hash_code()) {
        convert<uint64_t, T>(v1, v2);
    } else if (v1.type_hash == typeid(float32_t).hash_code()) {
        convert<float32_t, T>(v1, v2);
    } else if (v1.type_hash == typeid(float64_t).hash_code()) {
        convert<float64_t, T>(v1, v2);
    } else {
        assert(0);
    }
}

static inline void convert_from(const vector &v1, indx_vector &v2) {
    convert_from<uint32_t>(v1, v2);
}

} // namespace vector
} // namespace XAMG
