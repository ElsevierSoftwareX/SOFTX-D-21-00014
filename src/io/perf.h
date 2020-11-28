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

#include "logout.h"

namespace XAMG {

extern logout out;

struct perf_info {

  private:
    bool collect;

    uint32_t _mem_read;
    uint32_t _mem_write;
    uint32_t _flop;

    uint32_t _allreduce;
    uint32_t _iallreduce;

  public:
    perf_info() { reset(); }

    void reset() {
        collect = false;
        _mem_read = 0;
        _mem_write = 0;

        _flop = 0;

        _allreduce = 0;
        _iallreduce = 0;
    }

    void start() { collect = true; }

    void stop() { collect = false; }

    bool active() const { return collect; }

    void mem_read(const size_t val) {
        if (active())
            _mem_read += val;
    }

    void mem_write(const size_t val) {
        if (active())
            _mem_write += val;
    }

    void flop(const size_t val) {
        if (active())
            _flop += val;
    }

    void allreduce(const size_t val) {
        if (active())
            _allreduce += val;
    }

    void iallreduce(const size_t val) {
        if (active())
            _iallreduce += val;
    }

    void print() {
        XAMG::out << XAMG::DBG << XAMG::SUMMARY << "Single loop stats: " << std::endl;
        XAMG::out << XAMG::DBG << XAMG::SUMMARY << "\tVector reads: " << _mem_read << std::endl;
        XAMG::out << XAMG::DBG << XAMG::SUMMARY << "\tVector writes: " << _mem_write << std::endl;
        XAMG::out << XAMG::DBG << XAMG::SUMMARY << "\tVector FLOP: " << _flop << std::endl;
        XAMG::out << XAMG::DBG << XAMG::SUMMARY << "\tAllreduce op.: " << _allreduce << std::endl;
        XAMG::out << XAMG::DBG << XAMG::SUMMARY << "\tIallreduce op.: " << _iallreduce << std::endl;
    }

    void print_mem_usage() {
        XAMG::out << XAMG::DBG << "\tVector reads: " << _mem_read
                  << "\tVector writes: " << _mem_write << std::endl;
    }
};

} // namespace XAMG
