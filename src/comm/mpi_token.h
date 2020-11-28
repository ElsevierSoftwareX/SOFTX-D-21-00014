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

namespace XAMG {
namespace mpi {

struct mpi_request_placeholder {
    mpi_request_placeholder() {
        space[0] = 0;
        space[1] = 0;
    }
    mpi_request_placeholder(uint64_t x0, uint64_t x1) {
        space[0] = x0;
        space[1] = x1;
    }
    uint64_t space[2]; // 2x64 bits is supposed to be enough to hold a MPI_Request structure.
                       // If not enough, a static_assert should trigger the situation.
    bool operator==(const mpi_request_placeholder &other) {
        return (this->space[0] == other.space[0]) && (this->space[1] == other.space[1]);
    }
    bool operator!=(const mpi_request_placeholder &other) { return !operator==(other); }

    // NOTE: this prevents copying of this object
  private:
    mpi_request_placeholder(const mpi_request_placeholder &);
    mpi_request_placeholder &operator=(const mpi_request_placeholder &);
};

typedef mpi_request_placeholder token;
static token null_token{0, 0};

struct mpi_requests_placeholder {
    size_t n;
    uint64_t *space;
    mpi_requests_placeholder() = delete;
    mpi_requests_placeholder(size_t _n);
    ~mpi_requests_placeholder();
    mpi_request_placeholder &operator[](size_t elem);

    // NOTE: this prevents copying of this object
  private:
    mpi_requests_placeholder(const mpi_requests_placeholder &);
    mpi_requests_placeholder &operator=(const mpi_requests_placeholder &);
};
typedef mpi_requests_placeholder tokens;

////////////////////

struct token_ptr {

    XAMG::mpi::token *tok;

    token_ptr() : tok(nullptr) {}
    ~token_ptr() { free_token(); }

    XAMG::mpi::token &get_token() {
        if (tok == nullptr)
            alloc_token();

        return *tok;
    }

    void alloc_token() {
        assert((tok == nullptr) && "Reallocation of token objects is not allowed");
        tok = new XAMG::mpi::token;
    }

    void free_token() {
        if (tok != nullptr)
            delete tok;
        tok = nullptr;
    }
};

struct tokens_ptr {

    XAMG::mpi::tokens *toks;
    uint32_t ntoks;

    tokens_ptr() : toks(nullptr), ntoks(0) {}
    ~tokens_ptr() { free_tokens(); }

    const uint32_t &get_num_tokens() { return ntoks; }

    void set_num_tokens(const uint32_t _ntoks) {
        assert(toks == nullptr);
        ntoks = _ntoks;
    }

    XAMG::mpi::tokens &get_tokens() {
        if (toks == nullptr)
            alloc_tokens();

        return *toks;
    }

    XAMG::mpi::token &get_token(uint32_t i) {
        assert(i < ntoks);
        if (toks == nullptr)
            alloc_tokens();

        return (*toks)[i];
    }

    void alloc_tokens() {
        assert((toks == nullptr) && "Reallocation of token objects is not allowed");
        toks = new XAMG::mpi::tokens(ntoks);
    }

    void alloc_tokens(const uint32_t _ntoks) {
        set_num_tokens(_ntoks);
        alloc_tokens();
    }

    void free_tokens() {
        if (toks != nullptr)
            delete toks;
        toks = nullptr;
        ntoks = 0;
    }
};

} // namespace mpi
} // namespace XAMG
