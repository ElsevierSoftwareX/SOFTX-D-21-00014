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
#include "sys/timer.h"

extern ID id;

namespace XAMG {

struct time_monitor_object {
    bool enabled;
    bool active;
    float64_t t;

    float64_t timer;
    uint64_t cntr;

    time_monitor_object() {
        disable();
        reset();
    }
    void reset() {
        t = 0;
        timer = 0;
        cntr = 0;
        active = false;
    }

    void enable() { enabled = true; }
    void disable() { enabled = false; }
    void start() {
        assert(active == false);
        t = sys::timer();
        ++cntr;
        active = true;
    }
    void stop() {
        assert(active == true);
        timer += sys::timer() - t;
        active = false;
    }
};

struct time_monitor {
    std::map<std::string, time_monitor_object> list;

    time_monitor() {}

    void add(const std::string &key) {
#ifdef XAMG_MONITOR
        if (list.find(key) == list.end()) {
            list.emplace(std::make_pair(key, time_monitor_object()));
            // XAMG::out << "Adding key " << key << " to the monitor list" << std::endl;
        }
#endif
    }

    void reset() {
#ifdef XAMG_MONITOR
        for (auto &obj : list)
            obj.second.reset();
#endif
    }

    void enable() {
#ifdef XAMG_MONITOR
        for (auto &obj : list)
            obj.second.enable();
#endif
    }

    void disable() {
#ifdef XAMG_MONITOR
        for (auto &obj : list)
            obj.second.disable();
#endif
    }

    void start(const std::string &key) {
#ifdef XAMG_MONITOR
        if (list.find(key) != list.end()) {
            list[key].start();
        }
#endif
    }

    void stop(const std::string &key) {
#ifdef XAMG_MONITOR
        if (list.find(key) != list.end()) {
            list[key].stop();
        }
#endif
    }

    void activate_group(const std::string &grp) {
#ifdef XAMG_MONITOR
        if (grp == "hsgs") {
            add("hsgs_irecv");
            add("hsgs_isend");
            add("hsgs_diag");
            add("hsgs_core_offd");
            add("hsgs_numa_offd");
            add("hsgs_node_offd");
            add("hsgs_loop");
            add("hsgs_fin_comm");
            add("hsgs_serv");
        } else if (grp == "main") {
            add("alloc");
            add("params");
        } else if (grp == "node_recv") {
            add("node_recv_wait");
            add("node_recv_bcast");
            add("node_recv_replicate");
            add("node_recv_barrier");
        }
#endif
    }

    void print() {
#ifdef XAMG_MONITOR
        // mpi::barrier();
        // usleep(id.gl_proc * 1e3);
        // output of proc #0 is typically enough to estimate time losses
        // XAMG::out << XAMG::ALLRANKS << "MONITOR: proc " << id.gl_proc

        XAMG::out << XAMG::ALLRANKS << XAMG::LOG << "MONITOR: proc " << id.gl_proc << std::endl;

        for (auto &obj : list) {
            // XAMG::out << " || " << obj.first << " : " << obj.second.timer << "(" << obj.second.cntr << ")";
            XAMG::out << XAMG::ALLRANKS << XAMG::LOG << "\t\t" << obj.first << " : "
                      << obj.second.timer << " s. (" << obj.second.cntr << ")" << std::endl;
        }
        // XAMG::out << std::endl;
        // mpi::barrier();
#endif
    }
};

} // namespace XAMG
