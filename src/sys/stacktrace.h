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

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

namespace stacktrace {

struct entry {
    entry() : line(0) {}
    std::string file;
    size_t line;
    std::string function;
    std::string to_string() const {
        std::ostringstream os;
        os << file << " (" << line << "): " << function;
        return os.str();
    }
};

class call_stack {
  public:
    // num_discard - number of stack entries to discard at the top.
    call_stack(const size_t num_discard = 0);
    virtual ~call_stack() throw();
    std::string to_string() const {
        std::ostringstream os;
        for (size_t i = 0; i < stack.size(); i++)
            os << stack[i].to_string() << std::endl;
        return os.str();
    }
    std::vector<entry> stack;
};

} // namespace stacktrace

#ifdef __GNUC__

#include <stdio.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <stdlib.h>

namespace stacktrace {

constexpr size_t MAX_DEPTH = 32;
call_stack::call_stack(const size_t num_discard /*= 0*/) {
    using namespace abi;
    void *trace[MAX_DEPTH];
    int stack_depth = backtrace(trace, MAX_DEPTH);
    for (int i = num_discard + 1; i < stack_depth; i++) {
        Dl_info dlinfo;
        if (!dladdr(trace[i], &dlinfo))
            break;
        const char *symname = dlinfo.dli_sname;
        int status;
        char *demangled = abi::__cxa_demangle(symname, NULL, 0, &status);
        if (status == 0 && demangled)
            symname = demangled;
        // store entry to stack
        if (dlinfo.dli_fname && symname) {
            entry e;
            e.file = dlinfo.dli_fname;
            e.line = 0; // unsupported
            e.function = symname;
            stack.push_back(e);
        } else {
            entry e;
            e.file = dlinfo.dli_fname;
            e.line = 0; // unsupported
            e.function = "???";
            stack.push_back(e);
        }
        if (demangled)
            free(demangled);
    }
}

call_stack::~call_stack() throw() {}

} // namespace stacktrace
#else
namespace stacktrace {

call_stack::call_stack(const size_t num_discard /*= 0*/) {}

call_stack::~call_stack() throw() {}

} // namespace stacktrace
#endif // __GNUC__

namespace stacktrace {
void print(const std::string &header, int N) {
    stacktrace::call_stack st(N);
    std::cerr << header << std::endl;
    std::cerr << "Stack trace:" << std::endl;
    std::cerr << "-------------------------------------------" << std::endl;
    std::cerr << st.to_string();
    std::cerr << "-------------------------------------------" << std::endl << std::endl;
}
} // namespace stacktrace

/*
 * Simple test case
 *
 * NOTE: only works if all the symbols in the executable are exported. Use -rdynamic or
-Wl,--export-dynamic
 * compiler options. Tested only with GCC.
 *
 *
int bar(int x) {
    stacktrace::print();
    return 0;
}

void foo()
{
  bar(6);
}

int main()
{
    foo();
    return 0;
}
*/
