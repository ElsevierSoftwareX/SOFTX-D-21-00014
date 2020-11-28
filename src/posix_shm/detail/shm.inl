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
namespace shm {

namespace helper {
const uint32_t max_retry_attempts = 100000;

void *shm_map(void *addr, size_t len, int prot, int flags, int fd, uint64_t offset) {
    void *ptr;
    uint32_t retry_cntr(0);

    do {
        ptr = mmap(addr, len, prot, flags, fd, offset);
        ++retry_cntr;
    } while ((ptr == MAP_FAILED) && (errno == EAGAIN) && (retry_cntr < max_retry_attempts));

    if (ptr == MAP_FAILED) {
        XAMG::out << XAMG::ALLRANKS << "map error: " << errno
                  << " | mmap reiterations: " << retry_cntr << std::endl;
        XAMG::out << XAMG::ALLRANKS << "Mapping of shared memory region (mmap) failed..."
                  << std::endl;
        assert(0);
    }
    return ptr;
}

void shm_unmap(void *ptr, size_t size) {
    int ret;
    uint32_t retry_cntr(0);

    do {
        ret = munmap(ptr, size);
        ++retry_cntr;
    } while ((ret == -1) && (errno == EAGAIN) && (retry_cntr < max_retry_attempts));

    if (ret == -1) {
        XAMG::out << XAMG::ALLRANKS << "unmap error: " << errno
                  << " | munmap reiterations: " << retry_cntr << std::endl;
        XAMG::out << XAMG::ALLRANKS << "Unmapping of shared memory region (munmap) failed..."
                  << std::endl;
        assert(0);
    }
}

void shm_allocate(int fd, uint64_t offset, uint64_t len) {
    int ret;
    uint32_t retry_cntr(0);

    do {
        ret = posix_fallocate(fd, offset, len);
        ++retry_cntr;
    } while ((ret == EINTR) && (retry_cntr < max_retry_attempts));

    if (ret != 0) {
        XAMG::out << XAMG::ALLRANKS << "alloc error: " << ret
                  << " | alloc reiterations: " << retry_cntr << std::endl;
        XAMG::out << XAMG::ALLRANKS
                  << "Allocation of shared memory region (posix_fallocate) failed..." << std::endl;
        assert(0);
    }
}
} // namespace helper

struct shm_object {
    std::string handle;
    void *ptr;
    size_t size;

    shm_object(const size_t _size, const bool owner, mem::sharing sharing_mode) {
        mpi::comm_pool comm_group(mpi::intra_layer_comm(intra_layer_by_sharing_mode(sharing_mode)));
        handle = "/xamg_shm." + std::to_string(id.exec_id);

        switch (sharing_mode) {
        case mem::NUMA: {
            handle += ".nd." + std::to_string(id.gl_node) + ".nm." + std::to_string(id.nd_numa) +
                      ".id." + std::to_string(id.numa_shm_id);
            ++id.numa_shm_id;

            break;
        }
        case mem::NODE: {
            handle += ".nd." + std::to_string(id.gl_node) + ".id." + std::to_string(id.node_shm_id);
            ++id.node_shm_id;
            break;
        }
        default: {
            assert(0);
            break;
        }
        }

        int flags = O_CREAT | O_RDWR;

        if (owner)
            flags = flags | O_TRUNC;
        else
            mpi::barrier(comm_group.comm);

        int shm = shm_open(handle.c_str(), flags, 0777);
        assert(shm != -1);

        if (owner) {
            size = _size;
            helper::shm_allocate(shm, 0, size);
        } else {
            struct stat buf;
            int stat = fstat(shm, &buf);
            assert(stat != -1);
            size = buf.st_size;
        }

        ptr = helper::shm_map(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm, 0);

        if (ptr == MAP_FAILED) {
            if (size)
                XAMG::out << XAMG::ALLRANKS << "Shared memory allocation error for block of size "
                          << size << std::endl;

            ptr = nullptr;
        }

        if (owner)
            mpi::barrier(comm_group.comm);
        close(shm);
    }

    ~shm_object() {
        helper::shm_unmap(ptr, size);
        shm_unlink(handle.c_str());
    }

    template <typename T>
    T *get_ptr() {
        return (T *)ptr;
    }
};

} // namespace shm
} // namespace XAMG
