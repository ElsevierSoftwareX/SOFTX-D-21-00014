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

#include "xamg_types.h"
#include "xamg_headers.h"

#include <sys/types.h>
#include <sys/mman.h>

#include "posix_shm/shm.h"

extern ID id;

namespace XAMG {
namespace mem {

#ifndef XAMG_SEPARATE_OBJECT
std::map<void *, shm::shm_object *> shm_list;
#else
extern std::map<void *, shm::shm_object *> shm_list;
#endif

static inline void *alloc_shm(uint64_t size, uint16_t numa_id_owner, mem::sharing sharing_mode) {
    //    assert(numa_id_owner >= 0);
    bool owner = false;
    if ((id.nd_numa == numa_id_owner) && (!id.nm_core))
        owner = true;

    shm::shm_object *shared = new shm::shm_object(size, owner, sharing_mode);
    shm_list.emplace(std::make_pair(shared->ptr, shared));
    return shared->ptr;
}

static inline void free_shm(void *ptr) {
    auto elem = shm_list.find(ptr);
    if (elem == shm_list.end()) {
        assert(0 && "attempt to erase already erased SHM element");
    } else {
        delete elem->second;
    }

    shm_list.erase(ptr);
}

static inline void *get_aligned(void *buf) {
    void *ptr;

    if (!((uintptr_t)buf % XAMG_ALIGN_SIZE))
        ptr = buf;
    else
        ptr = (void *)((uintptr_t)buf + XAMG_ALIGN_SIZE - ((uintptr_t)buf % XAMG_ALIGN_SIZE));

    return ptr;
}

static inline bool if_aligned(const void *ptr) {
    if (((uintptr_t)ptr % XAMG_ALIGN_SIZE))
        return false;
    else
        return true;
}

static inline void *alloc_buffer(uint64_t size, mem::sharing sharing_mode,
                                 uint16_t numa_id_owner = 0) {
    void *ptr = nullptr;
    uint64_t aligned_size = size + (XAMG_ALIGN_SIZE - 1);

    switch (sharing_mode) {
    case mem::CORE: {
        ptr = malloc(aligned_size);
        break;
    }
    case mem::NUMA:
    case mem::NODE: {
        ptr = alloc_shm(aligned_size, numa_id_owner, sharing_mode);
        break;
    }
    default: {
        assert("incorrect sharing mode allocation" && 0);
    }
    }

    return ptr;
}

static inline void free_buffer(void *ptr, mem::sharing sharing_mode) {

    switch (sharing_mode) {
    case mem::CORE: {
        free(ptr);
        break;
    }
    case mem::NUMA:
    case mem::NODE: {
        free_shm(ptr);
        break;
    }
    default: {
        assert(0);
    }
    }
}

} // namespace mem
} // namespace XAMG
