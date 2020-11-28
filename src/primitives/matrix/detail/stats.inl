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
namespace matrix {

struct indicator_stats {
    uint32_t min = 0;
    uint32_t max = 0;
    float64_t avg = 0.0001234567e0;
};

struct matrix_stats {
    uint64_t nrows;    // global number of rows in the matrix
    uint64_t ncols;    // global number of columns in the matrix
    uint64_t nonzeros; // global number of nonzeros in the matrix

    indicator_stats nnz_per_row;
    indicator_stats comm_nbrs;
    indicator_stats comm_volume;

    matrix_stats() : nrows(0), ncols(0), nonzeros(0) {}

    void print_line() {
        XAMG::out << XAMG::LOG << std::right << std::setfill('-') << "|" << std::setw(131) << "|"
                  << std::endl;
    }

    void print_header() {
        print_line();

        XAMG::out << XAMG::LOG << std::setfill(' ') << std::left << "| " << std::setw(6) << "Info"
                  << " | " << std::setw(38) << "matrix"
                  << " | " << std::setw(24) << "nnz per row"
                  << " | " << std::setw(24) << "comm proc nbrs"
                  << " | " << std::setw(24) << "comm volume, elems"
                  << " |" << std::endl;
        XAMG::out << XAMG::LOG << std::setfill(' ') << std::left << "| " << std::setw(6) << ""
                  << " | " << std::setw(10) << "nrows"
                  << " | " << std::setw(10) << "ncols"
                  << " | " << std::setw(12) << "nonzeros";
        XAMG::out << std::setfill(' ') << std::left << " | " << std::setw(6) << "min"
                  << " | " << std::setw(6) << "max"
                  << " | " << std::setw(6) << "avg";
        XAMG::out << std::setfill(' ') << std::left << " | " << std::setw(6) << "min"
                  << " | " << std::setw(6) << "max"
                  << " | " << std::setw(6) << "avg";
        XAMG::out << std::setfill(' ') << std::left << " | " << std::setw(6) << "min"
                  << " | " << std::setw(6) << "max"
                  << " | " << std::setw(6) << "avg";
        XAMG::out << " |" << std::endl;

        print_line();
    }

    void print_footer() { print_line(); }

    void print_row(const std::string &str) {
        XAMG::out << std::defaultfloat;

        XAMG::out << XAMG::LOG << std::setfill(' ') << std::left << "| " << std::setw(6) << str
                  << " | " << std::setw(10) << nrows << " | " << std::setw(10) << ncols << " | "
                  << std::setw(12) << nonzeros << " | " << std::setw(6) << nnz_per_row.min << " | "
                  << std::setw(6) << nnz_per_row.max << " | " << std::setw(6)
                  << std::setprecision(3) << nnz_per_row.avg << " | " << std::setw(6)
                  << comm_nbrs.min << " | " << std::setw(6) << comm_nbrs.max << " | "
                  << std::setw(6) << std::setprecision(2) << comm_nbrs.avg << " | " << std::setw(6)
                  << comm_volume.min << " | " << std::setw(6) << comm_volume.max << " | "
                  << std::setw(6) << (size_t)comm_volume.avg << " |" << std::endl;

        XAMG::out << std::defaultfloat << std::setprecision(6);
    }

    void print(const std::string &str, const bool hierarchy = false) {
        if (!hierarchy) {
            print_header();
            print_row(str);
            print_footer();
        } else {
            print_row(str);
        }
    }
};

} // namespace matrix
} // namespace XAMG
