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

#ifndef XAMG_SEPARATE_OBJECT

vector::vector(const vector &that)
    : sharing_mode(that.sharing_mode), type_hash(that.type_hash), if_allocated(false),
      if_empty(that.if_empty), if_initialized(false), if_zero(true), ext_offset(that.ext_offset),
      nv(that.nv), active_numa_block(that.active_numa_block), vec_part(that.vec_part) {
    if (that.if_allocated) {
        if (type_hash == typeid(uint8_t).hash_code()) {
            alloc<uint8_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<uint8_t>();
            auto that_ptr = that.get_aligned_ptr<uint8_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(uint8_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(uint16_t).hash_code()) {
            alloc<uint16_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<uint16_t>();
            auto that_ptr = that.get_aligned_ptr<uint16_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(uint16_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(uint32_t).hash_code()) {
            alloc<uint32_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<uint32_t>();
            auto that_ptr = that.get_aligned_ptr<uint32_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(uint32_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(uint64_t).hash_code()) {
            alloc<uint64_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<uint64_t>();
            auto that_ptr = that.get_aligned_ptr<uint64_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(uint64_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(float32_t).hash_code()) {
            alloc<float32_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<float32_t>();
            auto that_ptr = that.get_aligned_ptr<float32_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(float32_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(float64_t).hash_code()) {
            alloc<float64_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<float64_t>();
            auto that_ptr = that.get_aligned_ptr<float64_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(float64_t);
            memcpy(ptr, that_ptr, vec_size);
        } else {
            assert(0);
        }

        if_zero = that.if_zero;
        if_initialized = that.if_initialized;
    } else {
        size = that.size;
        nv = that.nv;

        if_allocated = that.if_allocated;
    }
}

vector &vector::operator=(const vector &that) {
    assert(type_hash == that.type_hash);
    // assert(if_allocated);
    // assert(size == that.size);
    // assert(nv == that.nv);

    if (if_allocated && that.if_allocated) {
        assert(size == that.size);
        assert(nv == that.nv);
        assert(sharing_mode == that.sharing_mode);
    }

    if (that.if_allocated) {
        if (type_hash == typeid(uint8_t).hash_code()) {
            if (!if_allocated)
                alloc<uint8_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<uint8_t>();
            auto that_ptr = that.get_aligned_ptr<uint8_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(uint8_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(uint16_t).hash_code()) {
            if (!if_allocated)
                alloc<uint16_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<uint16_t>();
            auto that_ptr = that.get_aligned_ptr<uint16_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(uint16_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(uint32_t).hash_code()) {
            if (!if_allocated)
                alloc<uint32_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<uint32_t>();
            auto that_ptr = that.get_aligned_ptr<uint32_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(uint32_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(uint64_t).hash_code()) {
            if (!if_allocated)
                alloc<uint64_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<uint64_t>();
            auto that_ptr = that.get_aligned_ptr<uint64_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(uint64_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(float32_t).hash_code()) {
            if (!if_allocated)
                alloc<float32_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<float32_t>();
            auto that_ptr = that.get_aligned_ptr<float32_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(float32_t);
            memcpy(ptr, that_ptr, vec_size);
        } else if (type_hash == typeid(float64_t).hash_code()) {
            if (!if_allocated)
                alloc<float64_t>(that.size, that.nv);
            auto ptr = get_aligned_ptr<float64_t>();
            auto that_ptr = that.get_aligned_ptr<float64_t>();
            uint64_t vec_size = that.nv * that.size * sizeof(float64_t);
            memcpy(ptr, that_ptr, vec_size);
        } else {
            assert(0);
        }

        ext_offset = that.ext_offset;
    } else {
        assert(!if_allocated);
    }

    active_numa_block = that.active_numa_block;
    if_initialized = that.if_initialized;

    if_zero = that.if_zero;

    vec_part = that.vec_part;

    return *this;
}

#endif

template <typename T>
void vector::print(const std::string &str) const {
    //        check(nonempty);
    if (if_empty)
        return;
    check(initialized);

    uint64_t core_size, core_offset;
    get_core_range<T>(core_size, core_offset);
    uint64_t offset;

    if (sharing_mode == mem::NUMA_NODE)
        offset = core_offset + global_numa_offset();
    else
        offset = core_offset + ext_offset + numa_offset();

    // XAMG::out << XAMG::ALLRANKS << XAMG::VEC << "Proc sizes: " << core_size << " " << core_offset << std::endl;
    for (uint64_t i = 0; i < core_size; ++i) {
        XAMG::out << XAMG::ALLRANKS << XAMG::LOG << XAMG::VEC;
        XAMG::out.format("%s[%3ld] : | ", str.c_str(), i + offset);
        uint64_t ii = (i + core_offset) * nv;
        if (if_zero == true) {
            for (uint16_t m = 0; m < nv; ++m)
                XAMG::out << (T)0.0 << " | ";
        } else {
            for (uint16_t m = 0; m < nv; ++m) {
                XAMG::out << get_value<T>(ii + m) << " | ";
                // XAMG::out.format(" % .12e |", get_value<T>(ii + m));
            }
        }
        XAMG::out << std::endl;
    }
}

template <typename T>
void vector::get_core_range(uint64_t &core_size, uint64_t &core_offset) const {
    check(allocated);
    check(nonempty);

    switch (sharing_mode) {
    case mem::CORE: {
        core_offset = 0;
        core_size = size;
        break;
    }
    case mem::NUMA: {
        core_size = misc::split_range<T>(size, id.nm_ncores);
        core_offset = std::min(size, core_size * id.nm_core);
        core_size = std::min(core_size, size - core_offset);
        break;
    }
    case mem::NUMA_NODE: {
        assert(vec_part != nullptr);
        core_offset = vec_part->core_layer.block_indx[id.nm_core];
        core_size = vec_part->core_layer.block_size[id.nm_core];
        break;
    }
    case mem::NODE: {
        // perform by cores from  numa#0
        core_size = misc::split_range<T>(size, id.nm_ncores);
        core_offset = std::min(size, core_size * (id.nm_core + id.nd_numa * id.nm_ncores));
        core_size = std::min(core_size, size - core_offset);
        break;
    }
    default: {
        core_offset = core_size = 0; // to avoid compilation warnings
        assert(0);
        break;
    }
    }

    // XAMG::out << XAMG::ALLRANKS << XAMG::VEC << core_size << " || " << core_offset << std::endl;
}

template <typename I>
uint64_t search(const I *i_ptr, const uint32_t size, uint32_t elem) {
    if (size == 0)
        return 0;

    uint64_t ind = 0;
    uint32_t i1 = 0;
    uint32_t i2 = size;

    do {
        uint32_t i_new = (i1 + i2) / 2;

        if (i_ptr[i_new] < elem) {
            i1 = i_new;
        } else if (i_ptr[i_new] > elem) {
            i2 = i_new;
        } else {
            i1 = i2 = i_new;
        }
    } while (i2 - i1 > 1);

    if (i2 - i1 == 1) {
        ind = i2;
    }

    ind = i1;
    return ind;
}

template <typename I, typename T>
void vector::get_core_range(uint64_t &core_size, uint64_t &core_offset, const vector &indx) const {
    check(allocated);
    check(nonempty);
    uint32_t factor = XAMG_ALIGN_SIZE / sizeof(T);
    assert(factor);

    assert(sharing_mode == mem::NUMA_NODE);
    uint64_t basic_core_size, basic_core_offset;
    get_core_range<T>(basic_core_size, basic_core_offset);

    const I *XAMG_RESTRICT i_ptr = indx.get_aligned_ptr<I>();

    uint64_t r1 = 0;
    uint64_t r2 = 0;
    if (id.nm_ncores > 1) {
        // r1 = search<I>(i_ptr, indx.size, basic_core_offset);
        // r2 = search<I>(i_ptr, indx.size, basic_core_offset+basic_core_size);
        XAMG_VECTOR_ALIGN
        for (uint64_t i = 0; i < indx.size; ++i) {
            if (i_ptr[i] < basic_core_offset)
                r1++;
            if (i_ptr[i] < basic_core_offset + basic_core_size)
                r2++;
        }
        /*
                assert(r1 == rr1);
                assert(r2 == rr2);
        */
    } else {
        r1 = 0;
        r2 = indx.size;
    }
    //        XAMG::out << XAMG::ALLRANKS << XAMG::VEC << core_size << " || " << core_offset <<
    //        std::endl;
    core_offset = r1;
    core_size = r2 - r1;
}

// template<typename T, const uint16_t NV>
// void vector::print_L2_norm(const char *str) {
//    assert (str != NULL);
//
//    vector res;
//    res.alloc<T>(1, NV);
//
//////    blas::dot_global<T, NV>(this, this, res);
//    blas::dot_global<T, NV>(this, this, res);
//}

template <typename F>
void merge_col_blocks(const std::vector<vector> &blocks, vector &vec) {
    if (!blocks.size())
        return;

    uint64_t vec_size = 0;
    for (auto &block : blocks) {
        vec_size += block.size;
    }

    ////

    vec.alloc<F>(vec_size, blocks[0].nv);

    uint32_t cntr = 0;
    for (auto &block : blocks) {
        for (uint64_t i = 0; i < block.size; ++i) {
            vec.set_element<F>(cntr, block.get_element<F>(i));
            ++cntr;
        }
    }

    vec.if_zero = blocks[0].if_zero;
    vec.ext_offset = blocks[0].ext_offset;
}

template <typename F>
void split_by_rows(const vector &vec, std::vector<vector> &blocks, const part::part_layer &layer) {
    assert(vec.sharing_mode == mem::CORE);
    uint32_t nblocks = layer.nblocks;
    blocks.resize(nblocks);
    if (vec.if_empty)
        return;

    for (uint32_t nb = 0; nb < nblocks; ++nb) {
        blocks[nb].alloc<F>(layer.block_size[nb], vec.nv);

        for (uint32_t i = 0; i < layer.block_size[nb]; ++i) {
            blocks[nb].set_element<F>(i, vec.get_element<F>(i + layer.block_indx[nb]));
        }

        if (layer.block_size[nb]) {
            blocks[nb].if_empty = false;
            blocks[nb].if_zero = vec.if_zero;
            blocks[nb].if_initialized = true;
        }

        blocks[nb].ext_offset = vec.ext_offset + layer.block_indx[nb];
    }
}

} // namespace vector
} // namespace XAMG
