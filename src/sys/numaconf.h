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

#include <unistd.h>
#include <strings.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <stdint.h>

#include "comm/mpi_token.h"
#include "comm/mpi_wrapper.h"
#include "misc/misc.h"

#ifdef WITH_CUDA
#include "sys/gpuconf.h"
#endif

// TODO: Think about more elegant way to print this error message
#ifdef NDEBUG
#define XAMG_FATAL_ERROR(str)                                                                      \
    {                                                                                              \
        XAMG::out << XAMG::ALLRANKS << str << std::endl;                                           \
        XAMG::mpi::abort(str);                                                                     \
    }
#else
#define XAMG_FATAL_ERROR(str)                                                                      \
    { assert(0 && str); }
#endif

namespace XAMG {
namespace sys {

struct numa_conf {
    int nnumas = 0, ncores = 0, ngpus = 0;
    bool noautoconf = false;
    std::map<int, int> core_to_gpu;
    std::map<int, int> core_to_numa;
    std::vector<int> cores;
    mutable int core = -1;
    int nthreads = 0;
    void init_generic(); // gets ncores and ngpus from system
    void init_for_vals(int NC_NODE, int NN, int NG);
    void init_from_str(const std::string &str);
    int gpu_by_core(int core) const {
        auto gpuit = core_to_gpu.find(core);
        assert(gpuit != core_to_gpu.end());
        return gpuit->second;
    }
    int numa_by_core(int core) const {
        auto numait = core_to_numa.find(core);
        assert(numait != core_to_numa.end());
        return numait->second;
    }
};

static inline size_t getnumcores() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

static inline size_t getnumgpus() {
#ifdef WITH_CUDA
    return XAMG::CUDA::getnumgpus();
#else
    return 0;
#endif
}

#if 1
static inline bool setthreadaffinity(int n) {
    cpu_set_t my_set;
    CPU_ZERO(&my_set);
    CPU_SET(n, &my_set);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &my_set) == -1) {
        perror("sched_setaffinity");
        XAMG_FATAL_ERROR("sched_setaffinity failure");
    }
    return true;
}

static inline bool threadaffinityisset(int &nthreads, bool &ignored) {
    cpu_set_t mask;
    ignored = false;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        XAMG_FATAL_ERROR("sched_getaffinity failure");
    }
    int NC = sys::getnumcores();
    if (getenv("XAMG_IGNORE_AFFINITY")) {
        char *env = getenv("XAMG_IGNORE_AFFINITY");
        if (strlen(env) == 1 && *env == '1') {
            nthreads = NC;
            ignored = true;
            return false;
        }
    }
    int nset = 0;
    for (int i = 0; i < NC; i++) {
        nset += (CPU_ISSET(i, &mask) ? 1 : 0);
    }
    nthreads = nset;
    // We assume OK: exact one-to-one affinity or hyperthreading/SMT affinity for 2, 3 or 4 threads
    return nthreads > 0 && nthreads < 5 && nthreads != NC;
}

static inline int getthreadaffinity() {
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        XAMG_FATAL_ERROR("sched_getaffinity failure");
    }
    int core = -1;
    for (size_t i = 0; i < sizeof(mask) * 8; i++) {
        if (CPU_ISSET(i, &mask)) {
            core = (int)i;
            break;
        }
    }
    assert(core != -1);
    assert(core < (int)sys::getnumcores());
    return core;
}

#endif

#ifndef XAMG_SEPARATE_OBJECT
void numa_conf::init_generic() {
    core_to_gpu.clear();
    core_to_numa.clear();
    size_t NC = sys::getnumcores();
    size_t NG = sys::getnumgpus();
    for (size_t i = 0; i < NC; i++) {
        int G = -1;
        if (NG)
            G = i * NG / NC;
        core_to_gpu[i] = G;
        core_to_numa[i] = 0;
    }
    ncores = NC;
    nnumas = 1;
    ngpus = NG;
    return;
}

void numa_conf::init_for_vals(int NC_NODE, int NN, int NG) {
    core_to_gpu.clear();
    core_to_numa.clear();
    for (int i = 0; i < NC_NODE; i++) {
        int G = -1, N = 0;
        if (NG)
            G = i * NG / NC_NODE;
        if (NN)
            N = i * NN / NC_NODE;
        core_to_gpu[i] = G;
        core_to_numa[i] = N;
    }
    ncores = NC_NODE / NN;
    nnumas = NN;
    ngpus = NG;
    return;
}

void numa_conf::init_from_str(const std::string &str) {
    if (str.empty()) {
        init_generic();
        // TODO: revise to suppress this output from all procs
        XAMG::out << XAMG::DBG << "numaconf: initialized as generic, ncores=" << ncores
                  << " ngpus=" << ngpus << std::endl;
        return;
    }
    try {
        if (str[0] == 'n') {
            int ncores = -1, nnumas = 1, ngpus = -1;
            auto scheme = XAMG::misc::str_split(str, ':');
            if (scheme.empty()) {
                throw std::runtime_error("parse error");
            }
            for (auto &s : scheme) {
                if (s[0] != 'n') {
                    throw std::runtime_error("parse error");
                }
                auto kv = XAMG::misc::str_split(s, '=');
                if (kv.size() != 2) {
                    throw std::runtime_error("parse error");
                }
                if (kv[0] == "ncores") {
                    if (kv[1] == "*") {
                        ncores = sys::getnumcores();
                    } else {
                        ncores = std::stoi(kv[1]);
                        noautoconf = true;
                    }
                } else if (kv[0] == "nnumas") {
                    nnumas = std::stoi(kv[1]);
                } else if (kv[0] == "ngpus") {
                    if (kv[1] == "*") {
                        ngpus = sys::getnumgpus();
                    } else {
                        ngpus = std::stoi(kv[1]);
                    }
                }
            }
            if (ncores == -1)
                ncores = sys::getnumcores();
            if (ngpus == -1)
                ngpus = sys::getnumgpus();
            if (ncores <= 0 || nnumas <= 0 || ngpus < 0) {
                throw std::runtime_error("incorrect value");
            }
            init_for_vals(ncores * nnumas, nnumas, ngpus);
            XAMG::out << XAMG::DBG
                      << "numaconf: initialized from basic conf string, ncores=" << ncores
                      << " nnumas=" << nnumas << " ngpus=" << ngpus << std::endl;
            return;
        }
        auto s_numas = XAMG::misc::str_split(str, ';');
        nnumas = s_numas.size();
        size_t ngpus = 0;
        int numa = 0;
        for (auto &s_numa : s_numas) {
            std::vector<int> gpus;
            auto s_core_gpu = XAMG::misc::str_split(s_numa, '@');
            if (s_core_gpu.size() == 2) {
                auto s_gpus = XAMG::misc::str_split(s_core_gpu[1], ',');
                XAMG::misc::vstr_to_vint(s_gpus, gpus);
                assert(s_gpus.size() == gpus.size());
            }
            std::vector<int> local_cores;
            auto s_cores = XAMG::misc::str_split(s_core_gpu[0], ',');
            if (s_cores.size() == 1 && (!strncasecmp(s_cores[0].c_str(), "0x", 2))) {
                uint64_t mask = 0;
                long long n = std::stoll(s_cores[0], nullptr, 16);
                assert(n > 0 && n < (long long)UINT64_MAX + 1);
                mask = (uint64_t)n;
                for (int j = 0; j < 64; j++) {
                    if ((uint64_t)mask & ((uint64_t)1 << j)) {
                        local_cores.push_back(j);
                    }
                }
            } else {
                XAMG::misc::vstr_to_vint(s_cores, local_cores);
                assert(s_cores.size() == local_cores.size());
            }
            size_t NC = local_cores.size();
            size_t NG = gpus.size();
            for (size_t i = 0; i < NC; i++) {
                int G = -1;
                if (NG)
                    G = gpus[i * NG / NC];
                core_to_gpu[local_cores[i]] = G;
                core_to_numa[local_cores[i]] = numa;
            }
            numa++;
            ngpus += NG;
            ncores += NC;
            cores.insert(cores.end(), local_cores.begin(), local_cores.end());
        }
        XAMG::out << XAMG::DBG
                  << "numaconf: initialized from cpumask conf string, ncores=" << ncores
                  << " ngpus=" << ngpus << std::endl;
    } catch (std::runtime_error &ex) {
        XAMG::out << XAMG::WARN
                  << std::string("numaconf: handling/parsing conf string failed, falling back to "
                                 "generic: ") +
                         ex.what()
                  << std::endl;
        init_generic();
        XAMG::out << XAMG::DBG << "numaconf: initialized as generic, ncores=" << ncores
                  << " ngpus=" << ngpus << std::endl;
        return;
    } catch (...) {
        XAMG::out << XAMG::WARN
                  << "numaconf: handling/parsing conf string, falling back to generic."
                  << std::endl;
        init_generic();
        XAMG::out << XAMG::DBG << "numaconf: initialized as generic, ncores=" << ncores
                  << " ngpus=" << ngpus << std::endl;
        return;
    }
}
#endif

static inline const numa_conf &numa_conf_init(const std::string &str) {
    // FIXME global variable here, will fail in multithreaded
    static numa_conf conf;
    conf.init_from_str(str);
    int nthreads = 1;
    bool aff_ignored = false;
    bool aff_isset = threadaffinityisset(nthreads, aff_ignored);
    if (conf.noautoconf) {
        conf.core = -1;
    } else if (aff_isset) {
        conf.core = getthreadaffinity();
    } else {
#if 0
        // FIXME best option is to refuse work when no affinity
        if (!ignored) {
            XAMG_FATAL_ERROR("CPU affinity not set");
        }
#endif
        conf.core = -1;
    }
    conf.nthreads = nthreads;
    if (!aff_ignored && !aff_isset) {
        XAMG::out << XAMG::WARN << "CPU affinity seems to be not set! (nthreads=" << nthreads << ")"
                  << std::endl;
        XAMG::out << XAMG::WARN << "Suppose that all coreid == local_rank_num" << std::endl;
    }
    return conf;
}

static inline void numa_conf_check(const numa_conf &conf) {
    bool oversubscription = false;
    bool assorted = false;
    int core = conf.core;
    bool affinityisnotset = false;
    if (core == -1) {
        core = conf.core = id.nd_core;
        affinityisnotset = true;
    }
    int gpu = conf.gpu_by_core(core);
    int numa = conf.numa_by_core(core);
    mpi::comm_pool global_comm_group(mpi::GLOBAL);
    mpi::comm_pool node_comm_group(mpi::INTRA_NODE);
    XAMG::out << XAMG::DBG << XAMG::ALLRANKS << "numaconf: topo: ";
    XAMG::out << "rank=" << global_comm_group.proc << "/" << global_comm_group.nprocs << "; ";
    XAMG::out << "local=" << node_comm_group.proc << "/" << node_comm_group.nprocs << "; ";
    XAMG::out << "aff.core=" << core;
    if (conf.nthreads > 1) {
        XAMG::out << "[" << conf.nthreads << "]";
    }
    XAMG::out << " numanode=" << numa << " node=" << global_comm_group.proc / node_comm_group.nprocs
              << " gpu=" << gpu << std::endl;
    if (conf.nthreads > 1 && affinityisnotset) {
        XAMG::out << XAMG::WARN
                  << "numaconf: thread affinity is not one-to-one, may result in suboptimal "
                     "performance"
                  << std::endl;
    }
    if (conf.ngpus != (int)getnumgpus()) {
        XAMG::out << XAMG::WARN
                  << "numaconf: detected number of GPUs is different from given configuration, may "
                     "result in suboptimal performance:"
                  << conf.ngpus << "/" << getnumgpus() << std::endl;
    }
    std::vector<int> allcores;
    allcores.resize(node_comm_group.nprocs, -1);
    mpi::allgather<int>(&core, 1, &allcores[0], 1, node_comm_group.comm);
    int prev = -1;
    for (auto c : allcores) {
        if (c == -1) {
            XAMG_FATAL_ERROR("There are some not assigned cores in across the ranks");
        }
        if (c < prev) {
            assorted = true;
        }
        prev = c;
    }
    prev = -1;
    std::sort(allcores.begin(), allcores.end());
    for (auto c : allcores) {
        if (c == prev) {
            oversubscription = true;
        }
    }
    if (oversubscription) {
        XAMG::out << XAMG::WARN
                  << "numaconf: oversubscription detected, may result in suboptimal performance"
                  << std::endl;
        XAMG::out << XAMG::WARN
                  << "numaconf: if the oversubscription is intended, consider setting "
                     "XAMG_IGNORE_AFFINITY environment variable"
                  << std::endl;
    }
    if (assorted) {
        XAMG::out << XAMG::WARN
                  << "numaconf: MPI ranks within a node don't seem to reside on ascending cores, "
                     "may result in suboptimal performance"
                  << std::endl;
    }
    // basic cross-checks:
    if ((id.gl_nprocs % id.nd_ncores) || (id.gl_nprocs % id.nm_ncores)) {
        XAMG_FATAL_ERROR(
            "Total number of MPI processes do not match the number of ncores & nnumas specified");
    }
}

} // namespace sys
} // namespace XAMG
