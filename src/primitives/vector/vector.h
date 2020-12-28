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

#include "mem/mem.h"

#include "detail/backend.inl"

#include "comm/data_exchange.h"

#include "misc/misc.h"

#include "part/part.h"

/////////

namespace XAMG {
namespace vector {

struct state {
    size_t type_hash;

    bool if_allocated;   // flag handling memory allocation
    bool if_empty;       // flag indicating vector is allocated but empty
    bool if_initialized; // flag handling data initialization
    bool if_zero;        // flag handling zero data content (valid with if_initialized = true only!

    uint64_t size;
    uint64_t ext_offset;

    uint16_t nv;
    uint16_t active_numa_block; // to be removed!

    state()
        : type_hash(typeid(void).hash_code()), if_allocated(false), if_empty(true),
          if_initialized(false), if_zero(true), size(0), ext_offset(0), nv(0),
          active_numa_block(0) {}
};

static inline mem::sharing vector_sharing_mode(mem::allocation alloc_mode) {
    if (alloc_mode == mem::LOCAL) {
        return mem::CORE;
    }
    if (alloc_mode == mem::DISTRIBUTED) {
        return mem::NUMA_NODE;
    }
    if (alloc_mode == mem::SHARED) {
        assert("Sharing mode must be specified for the shared alloc mode" && 0);
    }

    return mem::CORE;
}

struct vector {

  private:
    //    state *_state;

  public:
    mem::sharing sharing_mode;

    size_t type_hash;

    // TODO: all flags must be placed in the shared memory!!!
    bool if_allocated;   // flag handling memory allocation
    bool if_empty;       // flag handling empty vector (vector is allocated of size = 0)
    bool if_initialized; // flag handling data initialization
    bool if_zero;        // flag handling zero data content (valid with if_initialized = true only!

    uint64_t size;
    uint64_t ext_offset;
    uint16_t nv;
    uint16_t active_numa_block;

    std::shared_ptr<part::part> vec_part;

    union {
        backend<int32_t> i32;
        backend<uint8_t> u8;
        backend<uint16_t> u16;
        backend<uint32_t> u32;
        backend<uint64_t> u64;
        backend<float32_t> f32;
        backend<float64_t> f64;
    } u;

    bool is_type_set() const { return type_hash != typeid(void).hash_code(); }
    template <typename T>
    void set_type() {
        assert(!is_type_set());
        type_hash = typeid(T).hash_code();
    }
    void set_type(size_t _type_hash) {
        assert(!is_type_set());
        type_hash = _type_hash;
    }
    template <typename T>
    backend<T> &get();
    template <typename T>
    const backend<T> &get() const;

    vector(mem::allocation alloc_mode_ = mem::LOCAL, mem::sharing sharing_mode_ = mem::CORE)
        : sharing_mode(sharing_mode_), type_hash(typeid(void).hash_code()), if_allocated(false),
          if_empty(true), if_initialized(false), if_zero(true), size(0), ext_offset(0), nv(0),
          vec_part(nullptr) {
        if (alloc_mode_ != mem::SHARED)
            sharing_mode = vector_sharing_mode(alloc_mode_);

        if (sharing_mode == mem::NUMA_NODE)
            active_numa_block = id.nd_numa;
        else
            active_numa_block = 0;
        //        _state = (state*) mem::alloc_buffer(sizeof(state), alloc_mode);
    }

    vector(mem::sharing sharing_mode_)
        : sharing_mode(sharing_mode_), type_hash(typeid(void).hash_code()), if_allocated(false),
          if_empty(true), if_initialized(false), if_zero(true), size(0), ext_offset(0), nv(0),
          vec_part(nullptr) {
        if (sharing_mode == mem::NUMA_NODE)
            active_numa_block = id.nd_numa;
        else
            active_numa_block = 0;
        //        _state = (state*) mem::alloc_buffer(sizeof(state), alloc_mode);
    }

    vector(const vector &that);
    vector &operator=(const vector &that);

    ~vector() {
        if (if_allocated) {
            if (type_hash == typeid(int32_t).hash_code())
                free_buf<int32_t>();
            else if (type_hash == typeid(uint8_t).hash_code())
                free_buf<uint8_t>();
            else if (type_hash == typeid(uint16_t).hash_code())
                free_buf<uint16_t>();
            else if (type_hash == typeid(uint32_t).hash_code())
                free_buf<uint32_t>();
            else if (type_hash == typeid(uint64_t).hash_code())
                free_buf<uint64_t>();
            else if (type_hash == typeid(float32_t).hash_code())
                free_buf<float32_t>();
            else if (type_hash == typeid(float64_t).hash_code())
                free_buf<float64_t>();
            else {
                assert(0);
            }
        }
        //        nv = 0;
        //        size = 0;
        //        ext_offset = 0;
        //        if_initialized = false;
        //        if_empty = true;
        //        if_allocated = false;
        //        if_zero = true;
    }

    void set_part(std::shared_ptr<part::part> part_) { vec_part = part_; }

    /////////
    /*

        size_t _type_hash() {
            return _state->type_hash;
        }

        bool _if_allocated() {
            return _state->if_allocated;
        }

        bool _if_initialized() {
            return _state->if_initialized;
        }

        bool _if_zero() {
            return _state->if_zero;
        }

        uint64_t _size() {
            return _state->size;
        }

        uint64_t _ext_offset() {
            return _state->ext_offset;
        }

        uint16_t _nv() {
            return _state->nv;
        }
    */

    /////////

    enum vector_status { unallocated, allocated, nonempty, uninitialized, initialized };

    void check(const vector_status status) const {
#ifdef XAMG_EXTRA_CHECKS
        assert(is_type_set());

        switch (status) {
        case unallocated:
            assert(!if_allocated);
            break;
        case allocated:
            assert(if_allocated);
            break;
        case nonempty:
            assert(if_allocated && !if_empty);
            break;
        case uninitialized:
            assert(if_allocated && !if_empty && !if_initialized);
            break;
        case initialized:
            assert(if_allocated && !if_empty && if_initialized);
            break;
        default:
            assert(0);
            break;
        }
#endif
    }

    template <typename T>
    void alloc(uint64_t block_size, const uint16_t _nv = 1) {
        if (!is_type_set())
            set_type<T>();
        else {
            assert(type_hash == typeid(T).hash_code());
        }

        check(unallocated);

        auto &vec_b = get<T>();
        nv = _nv;

        if (sharing_mode == mem::CORE) {
            vec_b.bufs = new void *[1];
            vec_b.ptrs = new T *[1];
            vec_b.buffer_size = new uint64_t[1];

            active_numa_block = 0;
            vec_b.buffer_size[active_numa_block] = block_size;
            vec_b.bufs[active_numa_block] = mem::alloc_buffer(
                vec_b.buffer_size[active_numa_block] * nv * sizeof(T), sharing_mode);
            vec_b.ptrs[active_numa_block] = (T *)mem::get_aligned(vec_b.bufs[active_numa_block]);

            size = vec_b.buffer_size[active_numa_block];
        } else if (sharing_mode == mem::NUMA) {
            vec_b.bufs = new void *[1];
            vec_b.ptrs = new T *[1];
            vec_b.buffer_size = new uint64_t[1];

            active_numa_block = 0;
            vec_b.buffer_size[active_numa_block] = block_size;
            vec_b.bufs[active_numa_block] = mem::alloc_buffer(
                vec_b.buffer_size[active_numa_block] * nv * sizeof(T), sharing_mode, id.nd_numa);
            vec_b.ptrs[active_numa_block] = (T *)(mem::get_aligned(vec_b.bufs[active_numa_block]));

            size = vec_b.buffer_size[active_numa_block];
        } else if (sharing_mode == mem::NODE) {
            vec_b.bufs = new void *[id.nd_nnumas];
            vec_b.ptrs = new T *[id.nd_nnumas];
            vec_b.buffer_size = new uint64_t[id.nd_nnumas];

            active_numa_block = 0;
            vec_b.buffer_size[active_numa_block] = block_size;
            vec_b.bufs[active_numa_block] = mem::alloc_buffer(
                vec_b.buffer_size[active_numa_block] * nv * sizeof(T), sharing_mode, 0);
            vec_b.ptrs[active_numa_block] = (T *)(mem::get_aligned(vec_b.bufs[active_numa_block]));

            // XAMG::out << XAMG::ALLRANKS << "NODE-Shared block size: " << block_size << " " << block_size * nv * sizeof(T) << std::endl;
            if (id.nd_numa) {
                vec_b.buffer_size[id.nd_numa] = block_size;
                vec_b.bufs[id.nd_numa] = mem::alloc_buffer(
                    vec_b.buffer_size[id.nd_numa] * nv * sizeof(T), mem::NUMA, id.nd_numa);
                vec_b.ptrs[id.nd_numa] = (T *)(mem::get_aligned(vec_b.bufs[id.nd_numa]));
            }

            size = vec_b.buffer_size[active_numa_block];
        } else if (sharing_mode == mem::NUMA_NODE) {
            vec_b.bufs = new void *[id.nd_nnumas];
            vec_b.ptrs = new T *[id.nd_nnumas];
            vec_b.buffer_size = new uint64_t[id.nd_nnumas];
            active_numa_block = id.nd_numa;
            vec_b.buffer_size[id.nd_numa] = block_size;

            if (!id.nm_core)
                mpi::allgather<uint64_t>(&block_size, 1, vec_b.buffer_size, 1, mpi::CROSS_NUMA);
            mpi::bcast<uint64_t>(vec_b.buffer_size, id.nd_nnumas, 0, mpi::INTRA_NUMA);

            for (uint16_t nm = 0; nm < id.nd_nnumas; ++nm) {
                vec_b.bufs[nm] =
                    mem::alloc_buffer(vec_b.buffer_size[nm] * nv * sizeof(T), mem::NODE, nm);
                vec_b.ptrs[nm] = (T *)mem::get_aligned(vec_b.bufs[nm]);
            }

            size = vec_b.buffer_size[active_numa_block];
        }
        if_allocated = true;
        if (size)
            if_empty = false;
    }

    template <typename T>
    void resize(const uint64_t &size_) {
        assert(sharing_mode == mem::CORE);

        auto &vec_b = get<T>();
        size = vec_b.buffer_size[0] = size_;
    }

    template <typename T>
    void free_buf() {
        auto vec_b = get<T>();

        switch (sharing_mode) {
        case mem::CORE: {
            mem::free_buffer(vec_b.bufs[0], mem::CORE);
            break;
        }
        case mem::NUMA: {
            mem::free_buffer(vec_b.bufs[0], mem::NUMA);
            break;
        }
        case mem::NODE: {
            mem::free_buffer(vec_b.bufs[0], mem::NODE);
            break;
        }
        case mem::NUMA_NODE: {
            for (uint16_t nm = 0; nm < id.nd_nnumas; ++nm)
                mem::free_buffer(vec_b.bufs[nm], mem::NODE);
            break;
        }
        default: {
            assert(0);
            break;
        }
        }

        delete[] vec_b.bufs;
        delete[] vec_b.ptrs;
        delete[] vec_b.buffer_size;
        vec_b.bufs = nullptr;
        vec_b.ptrs = nullptr;
        vec_b.buffer_size = nullptr;
    }

    // Returns size of all allocated data in bytes
    size_t get_elem_size() const {
        //        check(allocated);
        assert(is_type_set());

        size_t elem_size = 0;
        if (type_hash == typeid(int32_t).hash_code()) {
            elem_size = sizeof(int32_t);
        } else if (type_hash == typeid(uint8_t).hash_code()) {
            elem_size = sizeof(uint8_t);
        } else if (type_hash == typeid(uint16_t).hash_code()) {
            elem_size = sizeof(uint16_t);
        } else if (type_hash == typeid(uint32_t).hash_code()) {
            elem_size = sizeof(uint32_t);
        } else if (type_hash == typeid(uint64_t).hash_code()) {
            elem_size = sizeof(uint64_t);
        } else if (type_hash == typeid(float32_t).hash_code()) {
            elem_size = sizeof(float32_t);
        } else if (type_hash == typeid(float64_t).hash_code()) {
            elem_size = sizeof(float64_t);
        } else {
            assert(0 && "Unsupported data type");
        }
        return elem_size;
    }

    //    void set_active_numa(const uint32_t numa) const {
    //        check(allocated);
    //        auto this_ptr = const_cast<vector*> (this);
    //        this_ptr->active_numa_block = numa;
    //    }

    template <typename T>
    T *get_aligned_ptr(const int numa_block = -1) {
        assert(type_hash == typeid(T).hash_code());
        check(allocated);

        T *ptr;
        auto &vec_b = get<T>();
        if (numa_block == -1)
            ptr = (T *)vec_b.ptrs[active_numa_block];
        else {
            assert((numa_block >= 0) && (numa_block < id.nd_nnumas));
            ptr = (T *)vec_b.ptrs[numa_block];
        }

        assert(mem::if_aligned(ptr));
        return ptr;
        //        assert(mem::if_aligned(vec_b.ptrs[active_numa_block]));
        //        return (T*) vec_b.ptrs[active_numa_block];
    }

    template <typename T>
    const T *get_aligned_ptr(const int numa_block = -1) const {
        assert(type_hash == typeid(T).hash_code());
        check(allocated);

        assert(numa_block == -1);
        T *ptr;
        auto &vec_b = get<T>();
        if (numa_block == -1)
            ptr = (T *)vec_b.ptrs[active_numa_block];
        else {
            assert((numa_block >= 0) && (numa_block < id.nd_nnumas));
            ptr = (T *)vec_b.ptrs[numa_block];
        }

        assert(mem::if_aligned(ptr));
        return ptr;
        //        assert(mem::if_aligned(vec_b.ptrs[active_numa_block]));
        //        return (T*) vec_b.ptrs[active_numa_block];
    }

    template <typename T>
    void *get_buffer_ptr() {
        check(allocated);

        auto &vec_b = get<T>();
        return vec_b.bufs[active_numa_block];
    }

    template <typename T>
    const void *get_buffer_ptr() const {
        check(allocated);

        auto &vec_b = get<T>();
        return vec_b.bufs[active_numa_block];
    }

    template <typename T>
    T *alloc_and_get_aligned_ptr(const uint64_t size, const uint16_t nv = 1) {
        alloc<T>(size, nv);
        return (get_aligned_ptr<T>());
    }

    template <typename T>
    inline T get_value(const uint32_t i) {
        check(nonempty);
        check(initialized);

        if (if_zero)
            return 0.0;
        else
            return (get_aligned_ptr<T>()[i]);
    }

    template <typename T>
    inline void set_value(const uint32_t i, const T val) {
        check(nonempty);
        check(allocated);

        get_aligned_ptr<T>()[i] = val;

        if_empty = false;
        if_initialized = true;
    }

    template <typename T>
    inline const T get_value(const uint32_t i) const {
        check(nonempty);
        check(initialized);

        if (if_zero)
            return 0.0;
        else
            return (get_aligned_ptr<T>()[i]);
    }

    template <typename T>
    inline std::vector<T> get_element(const uint32_t i) const {
        check(nonempty);
        check(initialized);

        std::vector<T> elem(nv);

        if (if_zero) {
            for (uint32_t l = 0; l < nv; ++l)
                elem[l] = 0.0;
        } else {
            const T *ptr = get_aligned_ptr<T>();
            for (uint32_t l = 0; l < nv; ++l)
                elem[l] = ptr[i * nv + l];
        }

        return elem;
    }

    template <typename T>
    inline void set_element(const uint32_t i, const std::vector<T> &elem) {
        check(nonempty);
        check(allocated);
        assert(elem.size() == nv);

        T *ptr = get_aligned_ptr<T>();
        for (uint32_t l = 0; l < nv; ++l)
            ptr[i * nv + l] = elem[l];

        if_empty = false;
        if_initialized = true;
    }

    template <typename T>
    void push_to_buffer(comm::data_exchange_buffer &buf) const {
        assert(sharing_mode == mem::CORE);
        check(initialized);

        buf.push_scalar<bool>(if_zero);
        buf.push_scalar<uint64_t>(size);
        buf.push_scalar<uint64_t>(ext_offset);

        buf.push_scalar<uint16_t>(nv);
        buf.push_scalar<uint8_t>(sharing_mode);
        buf.push_scalar<size_t>(type_hash);

        /////////

        auto ptr = get_aligned_ptr<T>();
        buf.push_array<T>(ptr, size * nv);
    }

    template <typename T>
    void pull_from_buffer(comm::data_exchange_buffer &buf) {
        assert(sharing_mode == mem::CORE);
        // check(allocated);
        uint64_t _size;
        uint64_t _ext_offset;
        uint64_t _numa_block_offset;

        uint16_t _nv;
        uint8_t _sharing_mode;
        size_t _type_hash;

        buf.pull_scalar<bool>(if_zero);
        buf.pull_scalar<uint64_t>(_size);
        buf.pull_scalar<uint64_t>(_ext_offset);
        buf.pull_scalar<uint16_t>(_nv);
        buf.pull_scalar<uint8_t>(_sharing_mode);
        buf.pull_scalar<size_t>(_type_hash);

        assert(sharing_mode == _sharing_mode);

        if (if_allocated) {
            assert(nv == _nv);
            assert(size == _size);
            // assert(ext_offset == _ext_offset);
            assert(type_hash == _type_hash);
            ext_offset = _ext_offset;
        } else {
            alloc<T>(_size, _nv);
            ext_offset = _ext_offset;
            assert(type_hash == _type_hash);
        }

        /////////

        auto ptr = get_aligned_ptr<T>();
        buf.pull_array<T>(ptr, size * nv);
        if_initialized = true;
    }

    /////////

    template <typename T>
    void get_core_range(uint64_t &core_size, uint64_t &core_offset) const;

    template <typename I, typename T>
    void get_core_range(uint64_t &core_size, uint64_t &core_offset, const vector &indx) const;

    template <typename T>
    void print(const std::string &str) const;

    //    template<typename T, uint16_t NV>
    //    void print_L2_norm(const char *str);

    uint64_t numa_offset() const {
        uint64_t offset = 0;
        switch (sharing_mode) {
        case mem::CORE:
        case mem::NUMA: {
            offset = 0;
            break;
        }
        case mem::NUMA_NODE: {
            assert(vec_part != nullptr);
            offset = vec_part->numa_layer.block_indx[id.nd_numa];
            break;
        }
        case mem::NODE: {
            offset = 0;
            break;
        }
        default: {
            assert(0 && "offset undefined");
            break;
        }
        }

        return offset;
    }

    uint64_t node_offset() const {
        uint64_t offset = 0;

        if (sharing_mode == mem::NUMA_NODE) {
            assert(vec_part != nullptr);
            offset = vec_part->node_layer.block_indx[id.gl_node];
        } else if (sharing_mode == mem::CORE) {
            offset = ext_offset;
        } else {
            assert(0 && "NODE offset undefined");
        }
        return offset;
    }

    uint64_t intra_numa_offset() const {
        uint64_t offset = 0;
        if (sharing_mode == mem::NUMA_NODE) {
            assert(vec_part != nullptr);
            offset = vec_part->core_layer.block_indx[id.nm_core];
        } else if (sharing_mode == mem::CORE) {
            offset = 0;
        } else {
            assert(0 && "INTRA NUMA offset undefined");
        }

        return offset;
    }

    uint64_t intra_node_offset() const { return (numa_offset() + intra_numa_offset()); }

    uint64_t global_numa_offset() const { return (node_offset() + numa_offset()); }

    uint64_t global_core_offset() const { return (node_offset() + intra_node_offset()); }

    void sync_zero_flag() const {
#ifdef XAMG_EXTRA_CHECKS
        uint32_t flag = if_zero;
        uint32_t sum_flag = 0;
        mpi::allreduce_sum<uint32_t>(&flag, &sum_flag, 1);
        if ((sum_flag != 0) && (sum_flag != (uint32_t)id.gl_nprocs))
            XAMG::out << ALLRANKS << "flag state " << flag << " but global state is " << sum_flag
                      << std::endl;
        assert((sum_flag == 0) || (sum_flag == (uint32_t)id.gl_nprocs));
#endif
    }

    template <typename T>
    void replicate_buffer() {
        assert(sharing_mode == mem::NODE);
        if (id.nd_numa) {
            const auto nm0_ptr = get_aligned_ptr<T>(0);
            auto nm_ptr = get_aligned_ptr<T>(id.nd_numa);
            uint32_t vec_size = size * nv;

            if (vec_size > 10000) {
                uint32_t local_size = vec_size / id.nm_ncores;
                uint32_t local_off = local_size * id.nm_core;
                local_size = std::min(local_size, (uint32_t)(vec_size - local_off));

                memcpy(nm_ptr + local_off, nm0_ptr + local_off, local_size * sizeof(T));
            } else {
                if (id.numa_master_process()) {
                    memcpy(nm_ptr, nm0_ptr, vec_size * sizeof(T));
                }
            }
        }
    }
};

// NOTE: these functions must be instantiate only once, so all other object files
// may use XAMG_SEPARATE_OBJECT define to prevent linker errors.
#ifndef XAMG_SEPARATE_OBJECT
template <>
backend<int32_t> &vector::get<int32_t>() {
    return u.i32;
}
template <>
backend<uint8_t> &vector::get<uint8_t>() {
    return u.u8;
}
template <>
backend<uint16_t> &vector::get<uint16_t>() {
    return u.u16;
}
template <>
backend<uint32_t> &vector::get<uint32_t>() {
    return u.u32;
}
template <>
backend<uint64_t> &vector::get<uint64_t>() {
    return u.u64;
}
template <>
backend<float32_t> &vector::get<float32_t>() {
    return u.f32;
}
template <>
backend<float64_t> &vector::get<float64_t>() {
    return u.f64;
}

template <>
const backend<int32_t> &vector::get<int32_t>() const {
    return u.i32;
}
template <>
const backend<uint8_t> &vector::get<uint8_t>() const {
    return u.u8;
}
template <>
const backend<uint16_t> &vector::get<uint16_t>() const {
    return u.u16;
}
template <>
const backend<uint32_t> &vector::get<uint32_t>() const {
    return u.u32;
}
template <>
const backend<uint64_t> &vector::get<uint64_t>() const {
    return u.u64;
}
template <>
const backend<float32_t> &vector::get<float32_t>() const {
    return u.f32;
}
template <>
const backend<float64_t> &vector::get<float64_t>() const {
    return u.f64;
}
#endif

// TODO: check if constructor, destructor, copy-constructor, etc. are needed here
struct indx_vector : vector {
    uint64_t r1, r2;

    indx_vector(mem::sharing sharing_mode_) : vector(sharing_mode_), r1(0), r2(0) {}

    void alloc(uint64_t block_size) { vector::alloc<uint32_t>(block_size, 1); }
};

} // namespace vector
} // namespace XAMG

#include "detail/vector.inl"
#include "operations.h"
