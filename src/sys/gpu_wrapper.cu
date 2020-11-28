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

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include "xamg_types.h"
#include "io/logout.h"


#define CUDA_CALL(X) { cudaError_t err = X; if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); } }


namespace XAMG {
namespace CUDA {

static inline bool check_cc20(cudaDeviceProp &prop, std::string &reason) {
    bool ret = prop.major >= 2;
    if (!ret)
        reason += " compute capability check FAILED;";
    return ret;
}

static inline bool check_overlap(cudaDeviceProp &prop, std::string &reason) {
    bool ret = prop.deviceOverlap;
    if (!ret)
        reason += " overlap capability check FAILED;";
    return ret;

}
#if 0
static inline bool check_mode(cudaDeviceProp &prop, std::string &reason) {
    bool ret = prop.computeMode;
    if (!ret)
        reason += " compute mode check FAILED;";
    return ret;

}
#endif
static inline bool check_warpsize(cudaDeviceProp &prop, std::string &reason) {
    bool ret = prop.warpSize == 32;
    if (!ret)
        reason += " warp size check FAILED;";
    return ret;

}

struct device_traits {
    device_traits(int _id, const std::string _name, size_t _shared_mem, size_t _global_mem, size_t _mpc, bool _ecc,
                  size_t _clock, bool _uva) : 
                  id(_id), name(_name), shared_mem(_shared_mem), global_mem(_global_mem), mpc(_mpc), 
                  ecc(_ecc), clock(_clock), uva(_uva) {}
    int id;
    const std::string name;
    size_t shared_mem;
    size_t global_mem;
    size_t mpc;
    bool ecc;
    size_t clock;
    bool uva;
    std::vector<size_t> neighbours;
    bool operator==(device_traits &that) {
        bool equal = true;
        equal = equal && name == that.name;
        equal = equal && shared_mem == that.shared_mem;
        equal = equal && global_mem == that.global_mem;
        equal = equal && mpc == that.mpc;
        equal = equal && ecc == that.ecc;
        equal = equal && clock == that.clock;
        return equal;
    }
    bool operator!=(device_traits &that) {
        return !this->operator==(that);
    }
};

struct gpu_conf {
    std::vector<device_traits> devices;
    bool gpu_conf_done = false;
    bool p2p_already_enabled = false;
    uint64_t fingerprint = 0;
};

// FIXME global variable
gpu_conf *gpuconf = nullptr;
void gpu_conf_init();

int getnumgpus() {
    gpu_conf_init();
    if (!gpuconf->gpu_conf_done)
        return 0;
    return gpuconf->devices.size();
}

void setgpu(int i) {
    gpu_conf_init();
    if (!gpuconf->gpu_conf_done)
        return;
    int j = 0;
    for (auto &d : gpuconf->devices) {
        if (d.id == i)
            break;
        j++;
    }
    if (j == (int)gpuconf->devices.size()) {
        assert(false && "Device with given id is not configured");
        return;
    }
    CUDA_CALL(cudaSetDevice(gpuconf->devices[j].id));
    // return gpuconf->devices.size();
}

void gpu_conf_init() {
    if (gpuconf) {
        return;
    }
    gpuconf = new gpu_conf;
    gpu_conf &conf = *gpuconf;
    auto &devices = conf.devices;
    int n = 0;
    CUDA_CALL(cudaGetDeviceCount(&n));
    if (n == 0) {
        conf.gpu_conf_done = true;
        return;
    }
    for (int i = 0; i < n; i++) {
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, i));
        bool device_is_ok = true;
        std::string reason;
        device_is_ok = device_is_ok && check_cc20(prop, reason);
        device_is_ok = device_is_ok && check_overlap(prop, reason);
        device_is_ok = device_is_ok && check_warpsize(prop, reason);        
        if (device_is_ok) {
            bool device_is_uva = true;
            device_is_uva = device_is_uva && prop.unifiedAddressing;
            // for windows: device_is_uva = device_is_uva && prop.tccDriver;
            devices.push_back(
                device_traits(i, prop.name, prop.sharedMemPerBlock, prop.totalGlobalMem, 
                              prop.multiProcessorCount, prop.ECCEnabled, prop.clockRate, 
                              device_is_uva));
        } else {
            XAMG::out << XAMG::WARN << "GPU device dropped: name: " << prop.name << ":" << reason << std::endl;
        }
    }
    bool uniform = true;
    for (auto it = devices.begin(); it != devices.end(); ++it) {
        auto it2 = std::next(it);
        if (it2 != devices.end()) {
            if (*it != *it2) {
                uniform = false;
            }
        }
    }
    if (!uniform) {
        XAMG::out << XAMG::WARN << "gpuconf: GPUs in a system are different, non-uniform conf is not supported";
        conf.gpu_conf_done = true;
        return;
    }
    for (auto it = devices.begin(); it != devices.end(); ++it) {
        auto &dev = *it;
        if (!dev.uva)
            continue;
        CUDA_CALL(cudaSetDevice(dev.id));
        for (auto it2 = std::next(it); it2 != devices.end(); ++it2) {
            auto &dev2 = *it2;
            if (!dev2.uva)
                continue;
            int flag1 = 0, flag2 = 0;            
            CUDA_CALL(cudaDeviceCanAccessPeer(&flag1, dev.id, dev2.id));
            CUDA_CALL(cudaDeviceCanAccessPeer(&flag2, dev2.id, dev.id));
            dev.neighbours.push_back(it2 - devices.begin());
            dev2.neighbours.push_back(it - devices.begin());
            if (!conf.p2p_already_enabled) {
                CUDA_CALL(cudaDeviceEnablePeerAccess(dev2.id, 0));
            }
        }
    }
    conf.p2p_already_enabled = true;
    conf.gpu_conf_done = true;
}

}
}
