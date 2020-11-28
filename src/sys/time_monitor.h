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
#include "io/io.h"

extern ID id;

namespace XAMG {

struct time_monitor_object {
    bool enabled;
    float64_t t;

    float64_t timer;
    uint64_t cntr;

    time_monitor_object() {
        disable();
        reset();
    }
    void reset() {
#ifdef XAMG_MONITOR
        t = 0;
        timer = 0;
        cntr = 0;
#endif
    }

    void enable() {
#ifdef XAMG_MONITOR
        enabled = true;
#endif
    }
    void disable() {
#ifdef XAMG_MONITOR
        enabled = false;
#endif
    }
    void start() {
#ifdef XAMG_MONITOR
        t = io::timer();
        ++cntr;
#endif
    }
    void stop() {
#ifdef XAMG_MONITOR
        timer += io::timer() - t;
#endif
    }
};

struct time_monitor {
    time_monitor_object alloc;
    time_monitor_object params;

    time_monitor() {}
    void reset() {
        alloc.reset();
        params.reset();
    }

    void enable() {
        alloc.enable();
        params.enable();
    }

    void disable() {
        alloc.disable();
        params.disable();
    }

    void print() {
#ifdef XAMG_MONITOR
        // mpi::barrier();
        // usleep(id.gl_proc * 1e3);
        // output of proc #0 is typically enough to estimate time losses
        // XAMG::out << XAMG::ALLRANKS << "MONITOR: proc " << id.gl_proc
        XAMG::out << XAMG::LOG << "MONITOR: proc " << id.gl_proc << " || alloc : " << alloc.timer
                  << "(" << alloc.cntr << ")"
                  << " || params : " << params.timer << "(" << params.cntr << ")" << std::endl;
        // mpi::barrier();
#endif
    }
};

} // namespace XAMG
