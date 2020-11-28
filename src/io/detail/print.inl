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

/////////

namespace XAMG {

#ifndef XAMG_SEPARATE_OBJECT
void io::print_hline(const uint16_t nv) {
    uint16_t delimiter_len = 10 + nv * (2 * 15);

    XAMG::out << XAMG::CONVERGENCE;
    XAMG::out << std::setw(delimiter_len) << std::setfill('-');
    XAMG::out << " | " << std::endl;
}

void io::print_residuals_header(uint16_t res_type, const uint16_t nv) {
    print_hline(nv);

    XAMG::out << XAMG::CONVERGENCE;
    XAMG::out << std::setfill(' ') << std::setw(7) << "Iter"
              << " | ";

    for (uint16_t i = 0; i < nv; ++i)
        XAMG::out << std::setw(12) << "res"
                  << " | " << std::setw(12) << "res / res0"
                  << " | ";
    XAMG::out << std::endl;

    print_hline(nv);

    XAMG::out << XAMG::CONVERGENCE;
    XAMG::out << std::setfill(' ') << std::setw(7) << "NV"
              << " | ";
    for (uint16_t i = 0; i < nv; ++i)
        XAMG::out << std::setw(27) << i << " | ";
    XAMG::out << std::endl;

    print_hline(nv);
}

void io::print_residuals_footer(const uint16_t nv) {
    print_hline(nv);
}

void io::print_bits(uint16_t ch) {
    uint8_t nbits = 8 * sizeof(decltype(ch));
    std::string str("");

    for (uint8_t i = 0; i < nbits; ++i) {
        if ((ch >> i) & 1)
            str.insert(0, 1, '1');
        else
            str.insert(0, 1, '0');
    }

    //    XAMG::out << "ch = " << ch << std::endl;
    XAMG::out << "bitwise : " << str << std::endl;
}
#endif

template <typename T>
void io::print_residuals(const uint32_t iter, const vector::vector &res, const vector::vector &res0,
                         const vector::vector &conv) {
    XAMG::out << std::scientific;

    XAMG::out << XAMG::CONVERGENCE;
    XAMG::out << std::setfill(' ') << std::setw(7) << iter << " | ";

    std::vector<T> res_elem = res.get_element<T>(0);
    std::vector<T> res0_elem = res0.get_element<T>(0);
    std::vector<T> conv_elem = conv.get_element<T>(0);

    for (uint16_t nv = 0; nv < res.nv; nv++) {
        if (conv_elem[nv]) {
            XAMG::out << std::setw(12) << sqrt(res_elem[nv]) << " | " << std::setw(12)
                      << sqrt(res_elem[nv] / res0_elem[nv]) << " | ";
        } else {
            XAMG::out << std::setfill('-') << std::setw(12) << "-"
                      << " | " << std::setw(12) << "-"
                      << " | ";
        }
    }
    XAMG::out << std::endl;

    XAMG::out << std::defaultfloat;
}

#ifndef XAMG_SEPARATE_OBJECT
void io::print_matrix_block(const matrix::matrix_block &block) {
    std::vector<uint64_t> col;
    std::vector<float64_t> val;

    uint64_t core_size, core_offset;
    uint64_t row_offset = block.data->get_block_row_offset();
    uint64_t col_offset = block.data->get_block_col_offset();

    if (!block.data->indexed()) {
        for (uint64_t l = 0; l < block.data->get_nrows(); ++l) {
            block.data->get_row(l, col, val);

            XAMG::out << XAMG::ALLRANKS;
            XAMG::out.format("row %4ld : ", l + row_offset);

            for (uint64_t i = 0; i < col.size(); ++i)
                XAMG::out.format("(%ld: %.4e) ", col[i] + col_offset, val[i]);
            XAMG::out << std::endl;
        }
    } else {
        for (uint64_t l = 0; l < block.data->get_nrows(); ++l) {
            block.data->get_row(l, col, val);

            XAMG::out << XAMG::ALLRANKS;
            XAMG::out.format("row %4ld : ", block.data->unpack_row_indx(l) + row_offset);

            for (uint64_t i = 0; i < col.size(); ++i)
                XAMG::out.format("(%ld: %.4e) ", block.data->unpack_col_indx(col[i]) + col_offset,
                                 val[i]);
            XAMG::out << std::endl;
        }
    }
}

void io::print_matrix(const matrix::matrix &m, const std::string &str) {
    XAMG::out << XAMG::ALLRANKS << "<" << str << ">" << std::endl;
    XAMG::out << XAMG::ALLRANKS << "DIAG:" << std::endl;
    auto &numa_layer = m.data_layer.find(segment::NUMA)->second;
    auto &node_layer = m.data_layer.find(segment::NODE)->second;

    io::print_matrix_block(numa_layer.diag);

    for (uint32_t nb = 0; nb < numa_layer.offd.size(); ++nb) {
        XAMG::out << XAMG::ALLRANKS << "NUMA_OFFD " << nb << ":" << std::endl;
        io::print_matrix_block(numa_layer.offd[nb]);
    }

    for (uint32_t nb = 0; nb < node_layer.offd.size(); ++nb) {
        XAMG::out << XAMG::ALLRANKS << "NODE_OFFD " << nb << ":" << std::endl;
        io::print_matrix_block(node_layer.offd[nb]);
    }
    getchar();
}
#endif

template <typename T>
void io::print_backend_matrix(T &mat) {
    mat.print_matrix();
}

template <typename T, const uint16_t NV>
void logout::norm(const vector::vector &x, const std::string &s) {
    vector::vector res;
    res.alloc<T>(1, NV);
    XAMG::blas::dot_global<T, NV>(x, x, res);
    std::vector<T> res_elem = res.get_element<T>(0);
    *this << XAMG::LOG << XAMG::VEC << "||" << s << "|| = ";
    for (uint16_t nv = 0; nv < NV; ++nv) {
        format(" % .12e |", sqrt(res_elem[nv]));
    }
    *this << std::endl;
}

template <typename T>
void logout::vector(const vector::vector &x, const std::string &s) {
    x.print<T>(s);
}

template <typename T>
void logout::vector(const std::vector<T> &x, const std::string &s) {
    *this << XAMG::VEC << s << " : | ";
    for (size_t i = 0; i < s.size(); ++i) {
        format(" % .12e |", x[i]);
    }
    *this << std::endl;
}

} // namespace XAMG
