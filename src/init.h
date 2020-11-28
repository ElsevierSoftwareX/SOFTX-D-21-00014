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
#include "io/logout.h"
#include "sys/numaconf.h"
#include "sys/time_monitor.h"
#include <time.h>

ID id;
XAMG::time_monitor monitor;
XAMG::perf_info perf;

namespace XAMG {
logout out;
}

#ifdef XAMG_WITH_SEPARATE_OBJECTS
#include "comm/mpi_token.h"
#include "comm/mpi_wrapper.h"
#else
#include "comm/mpi_wrapper_inline.h"
#endif

/////////

namespace XAMG {

void init(int argc, char *argv[], const std::string &snumaconf = "",
          const std::string &logname = "") {
    if (logname.length())
        out.set_output(logname);
        // out.set_types(out_type_t::INFO | out_type_t::SUMMARY);
        // out.remove_types(out_type_t::VECTORS);
        // out.remove_types(out_type_t::INFO);
#ifdef XAMG_IO_DEBUG
    out.add_types(out_type_t::DEBUG);
#endif
    const char *env = getenv("XAMG_NUMACONF");
    if (env == NULL) {
        env = snumaconf.c_str();
    }
    auto &conf = sys::numa_conf_init(env);
    mpi::init(argc, argv, conf);
    sys::numa_conf_check(conf);

    mpi::set_exec_id();

    std::string prefix = "gmon_" + std::to_string(id.gl_proc) + ".out";
    setenv("GMON_OUT_PREFIX", prefix.c_str(), 0);
}

void finalize() {
    mpi::finalize();
}

} // namespace XAMG
