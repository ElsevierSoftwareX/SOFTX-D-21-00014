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
namespace mpi {

mpi_requests_placeholder::mpi_requests_placeholder(size_t _n) : n(_n), space(nullptr) {
    size_t s = sizeof(MPI_Request);
    space = (uint64_t *)malloc(s * n);
}

mpi_request_placeholder &mpi_requests_placeholder::operator[](size_t elem) {
    size_t s = sizeof(MPI_Request);
    char *p = (char *)space + s * elem;
    return *((mpi_request_placeholder *)p);
}

mpi_requests_placeholder::~mpi_requests_placeholder() {
    free(space);
}

MPI_Request *token2req(token &tok) {
    static_assert(sizeof(MPI_Request) <= sizeof(token),
                  "Size mpi_request_placeholder is too small to hold MPI_Reqest");
    return (MPI_Request *)&tok;
}

int get_num_tokens(tokens &toks) {
    return toks.n;
}

MPI_Request *tokens2reqptr(tokens &toks) {
    static_assert(sizeof(MPI_Request) <= sizeof(mpi_request_placeholder),
                  "Size mpi_request_placeholder is too small to hold MPI_Reqest");
    return (MPI_Request *)toks.space;
}

} // namespace mpi
} // namespace XAMG
