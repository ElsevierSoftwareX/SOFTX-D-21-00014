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

#include <ostream>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <regex>
#include <functional>
#include "misc/misc.h"
#include "xamg_types.h"

class ID;
extern ID id;

namespace XAMG {

namespace vector {
class vector;
}

struct logout;

struct logout_modifier {
    enum type_t {
        NONE = 0,
        INFO = 1,
        DEBUG = 2,
        VECTORS = 4,
        SUMMARY = 8,
        PARAMS = 16,
        WARN = 32,
        CONVERGENCE = 64,
        LOG = 128
    };

    static uint32_t strtotype(const std::string &s) {
        if (s == "NONE")
            return NONE;
        else if (s == "INFO")
            return INFO;
        else if (s == "DEBUG")
            return DEBUG;
        else if (s == "VECTORS")
            return VECTORS;
        else if (s == "SUMMARY")
            return SUMMARY;
        else if (s == "PARAMS")
            return PARAMS;
        else if (s == "WARN")
            return WARN;
        else if (s == "CONVERGENCE")
            return CONVERGENCE;
        else if (s == "LOG")
            return LOG;
        return 0;
    }

  protected:
    type_t type = NONE;
    bool suppress = false;

  public:
    virtual type_t get_type() const { return type; }
    virtual bool get_suppress() const { return suppress; }
    // Do some other action with logout besides changing type and suppress state.
    // Can be used to output additional prefix before message.
    virtual void action(logout &) const {};
};

struct logout_modifier_always_suppress : public logout_modifier {
    logout_modifier_always_suppress() { logout_modifier::suppress = true; }
};

template <logout_modifier::type_t TYPE>
struct logout_modifier_bytype : public logout_modifier {
    const std::string line_starter;
    logout_modifier_bytype(const std::string &_line_starter = "") : line_starter(_line_starter) {
        logout_modifier::type = TYPE;
    }
    // Implemented below since depends on logout class details
    virtual void action(logout &) const override;
};

struct logout_modifier_bylambda : public logout_modifier {
    std::function<bool()> const lmbd;
    logout_modifier_bylambda(std::function<bool()> const &_lmbd) : lmbd(_lmbd) {}
    virtual bool get_suppress() const override { return !lmbd(); }
};

using out_type_t = XAMG::logout_modifier::type_t;
#ifndef XAMG_SEPARATE_OBJECT
logout_modifier_bytype<out_type_t::INFO> INFO;
logout_modifier_bytype<out_type_t::DEBUG> DBG;
logout_modifier_bytype<out_type_t::VECTORS> VEC;
logout_modifier_bytype<out_type_t::SUMMARY> SUMMARY("SUMMARY:: ");
logout_modifier_bytype<out_type_t::PARAMS> PARAMS("PARAMS:: ");
logout_modifier_bytype<out_type_t::WARN> WARN("WARNING: ");
logout_modifier_bytype<out_type_t::CONVERGENCE> CONVERGENCE("CONV:: ");
logout_modifier_bytype<out_type_t::LOG> LOG("LOG:: ");

logout_modifier_always_suppress MUTE;
logout_modifier_bylambda ALLRANKS([]() -> bool { return true; });
logout_modifier_bylambda ZERO_RANK_ONLY([]() -> bool {
    return id.gl_proc == 0 || id.gl_proc == -1;
});
#else
extern logout_modifier_bytype<out_type_t::INFO> INFO;
extern logout_modifier_bytype<out_type_t::DEBUG> DBG;
extern logout_modifier_bytype<out_type_t::VECTORS> VEC;
extern logout_modifier_bytype<out_type_t::SUMMARY> SUMMARY;
extern logout_modifier_bytype<out_type_t::PARAMS> PARAMS;
extern logout_modifier_bytype<out_type_t::WARN> WARN;
extern logout_modifier_bytype<out_type_t::CONVERGENCE> CONVERGENCE;
extern logout_modifier_bytype<out_type_t::LOG> LOG;

extern logout_modifier_always_suppress MUTE;
extern logout_modifier_bylambda ALLRANKS;
extern logout_modifier_bylambda ZERO_RANK_ONLY;
#endif

logout &operator<<(XAMG::logout &l, const logout_modifier &mod);

struct logout : std::ostream, std::streambuf {
    using type_t = logout_modifier::type_t;
    constexpr static size_t fbsize = 1024;
    char format_buffer[fbsize];
    std::basic_ostream<char> *out = nullptr;
#ifdef XAMG_DEBUG
    uint32_t types_mask = 0xffffffff;
#else
    uint32_t types_mask = 0xffffffff & (~type_t::DEBUG);
#endif

    enum { NONE, STARTED, NEWLINE, SUPPRESS } state = NONE;
    bool out_created = false; // must be changed to 'true' when output stream was allocated with
                              // 'new' and requires 'delete'
    bool suppress_was_set = false,
         type_was_set = false; // flags for handling default behaviour (if no modifiers in string)
    bool force_zero_rank_only = false;
    logout_modifier_bylambda &default_suppress_mod = ZERO_RANK_ONLY;
    type_t default_type = type_t::INFO;
    const std::string line_starter = "XAMG: ";
    logout() : std::ostream(this) {
        out = &std::cout;
        get_types_mask();
    }
    logout(const std::string &name) : std::ostream(this) {
        set_output(name);
        if (!out) {
            out = &std::cout;
        }
        get_types_mask();
    }
    // Read XAMG_LOG environment if it is set. Comma separated list of flags.
    // Special flags: RANK0: forcely suppress any out to ranks != 0; ALL: turn on all flags
    // Other flags: are just type names from enum logout_modifier::type_t.
    // Prefix "^" means supressing the type.
    // Example: export XAMG_LOG="RANK0,ALL,^VECTORS"
    void get_types_mask() {
        uint32_t add_mask = 0, remove_mask = 0;
        char *env = getenv("XAMG_LOG");
        if (!env)
            return;
        auto strs = XAMG::misc::str_split(env, ',');
        for (auto &f : strs) {
            if (f == "ALL") {
                add_mask |= 0xffffffff;
                continue;
            }
            if (f == "RANK0") {
                force_zero_rank_only = true;
                continue;
            }
            if (std::regex_match(f, std::regex("\\^.*"))) {
                auto splt = XAMG::misc::str_split(f, '^');
                assert(splt.size() == 2);
                auto &remove_flag = splt[1];
                remove_mask |= logout_modifier::strtotype(remove_flag);
            } else {
                auto &add_flag = f;
                add_mask |= logout_modifier::strtotype(add_flag);
            }
        }
        types_mask = add_mask & (~remove_mask);
    }
    void set_types(uint32_t _types_mask) { types_mask = _types_mask; }
    void add_types(uint32_t _types_mask) { types_mask |= _types_mask; }
    void remove_types(uint32_t _types_mask) { types_mask &= (~_types_mask); }
    void set_output(const std::string &name) {
        assert(!out_created);
        auto fout = new std::ofstream();
        if (fout) {
            fout->open(name);
            if (!fout->is_open()) {
                delete fout;
            }
            out_created = true;
            out = fout;
        }
    }
    ~logout() {
        if (out) {
            out->flush();
            if (out_created)
                delete out;
        }
    }
    int overflow(int c) {
        handle(c);
        return 0;
    }
    void print_line_starter() {
        assert(out);
        (*out) << line_starter;
    }
    void handle(char c) {
        assert(out);
        if (state == SUPPRESS) {
            if (c == '\n') {
                suppress_was_set = type_was_set = false;
                state = NEWLINE;
            }
            return;
        }
        if (state == NONE || state == NEWLINE) {
            // Handle defaults for a new string
            if (!suppress_was_set || force_zero_rank_only) {
                *this << default_suppress_mod;
            }
            if (!type_was_set) {
                set_type(default_type);
            }
            if (state == SUPPRESS) {
                return;
            }
        }
        if (state == NONE || state == NEWLINE) {
            print_line_starter();
            state = STARTED;
        }
        if (c == '\n') {
            suppress_was_set = type_was_set = false;
            state = NEWLINE;
        }
        out->put(c);
    }
    int format(const char *format, ...) {
        va_list args;
        va_start(args, format);
        int r = vsnprintf(format_buffer, fbsize - 1, format, args);
        va_end(args);
        if (r > 0 && r < (int)(fbsize - 1)) {
            format_buffer[r + 1] = 0;
            *this << format_buffer;
        }
        return r;
    }
    void set_type(logout_modifier::type_t t) {
        type_was_set = true;
        if (((uint32_t)t & types_mask) == 0) {
            state = SUPPRESS;
        }
    }
    void set_suppress(bool s) {
        assert((force_zero_rank_only || !suppress_was_set) &&
               "You can change the suppress state only once at the beginning of a string");
        suppress_was_set = true;
        if (s) {
            state = SUPPRESS;
        }
    }
    // NOTE: implemented in io/detail/print.inl
    template <typename T, const uint16_t NV>
    void norm(const vector::vector &x, const std::string &s);

    // NOTE: implemented in io/detail/print.inl
    template <typename T>
    void vector(const vector::vector &x, const std::string &s);

    // NOTE: implemented in io/detail/print.inl
    template <typename T>
    void vector(const std::vector<T> &x, const std::string &s);
};

template <logout_modifier::type_t TYPE>
void logout_modifier_bytype<TYPE>::action(logout &l) const {
    if (line_starter.length())
        l << line_starter;
}

#ifndef XAMG_SEPARATE_OBJECT
logout &operator<<(logout &l, const logout_modifier &mod) {
    if (mod.get_type() != logout_modifier::NONE)
        l.set_type(mod.get_type());
    else
        l.set_suppress(mod.get_suppress());
    mod.action(l);
    return l;
}
#endif
} // namespace XAMG
