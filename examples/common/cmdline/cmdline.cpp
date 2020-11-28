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

// FIXME Think of more elegant way to do this
#define XAMG_SEPARATE_OBJECT

#include <iostream>
#include <fstream>
#include <regex>
#include <argsparser.h>
#include <xamg_headers.h>
#include <xamg_types.h>

#include <param/params.h>

#undef XAMG_SEPARATE_OBJECT

#include "cmdline.h"

extern ID id;

void parser_add_common(args_parser &parser, const std::vector<std::string> &solver_types,
                       XAMG::params::global_param_list &params) {
    // Add 'sys' options descriptions:
    // NOTE: SYS group is a group of options which are not read or saved into YAML-config
    parser.set_current_group("SYS");
    parser.add<std::string>("dump", "").set_caption(DUMP_CAPTION).set_description(DUMP_DESCR);
    parser.add<std::string>("load", "").set_caption(LOAD_CAPTION).set_description(LOAD_DESCR);
    parser.add<int>("sleep", 0).set_caption(SLEEP_CAPTION).set_description(SLEEP_DESCR);
    parser.set_default_current_group();

    // Add main options descriptions:
    parser.add<std::string>("matrix", "generate")
        .set_caption(MATRIX_CAPTION)
        .set_description(MATRIX_DESCR);
    parser.add<std::string>("node_config", "")
        .set_caption(NODE_CONFIG_CAPTION)
        .set_description(NODE_CONFIG_DESCR);
    for (auto &s : solver_types) {
        std::string opt = s + "_params";
        parser.add_map(opt.c_str(), "")
            .set_caption(SOLVER_PARAMS_CAPTION)
            .set_description(SOLVER_PARAMS_DESCR);
        if (std::regex_search(s, std::regex("smooth")) || s == "solver" || s == "preconditioner") {
            opt = s + "_override";
            parser.add_map(opt.c_str(), "")
                .set_caption(PARAMS_OVERRIDE_CAPTION)
                .set_description(PARAMS_OVERRIDE_DESCR);
        }
    }
    parser.add_map("generator_params", "")
        .set_caption(GENERATOR_PARAMS_CAPTION)
        .set_description(GENERATOR_PARAMS_DESCR);
    //    parser.add_flag("graph_reordering").set_caption(GRAPH_REORDERING_CAPTION).set_description(GRAPH_REORDERING_DESCR);
}

parse_result_t parse_cmdline_common(args_parser &parser,
                                    const std::vector<std::string> &solver_types,
                                    XAMG::params::global_param_list &params) {
    // Do actual command line parsing:
    if (!parser.parse()) {
        if (parser.is_help_mode())
            return PARSE_HELP_PRINTED;
        std::cerr << "Command line parse error. Use -help option for help." << std::endl;
        return PARSE_FATAL_FAILURE;
    }

    // Get parsing results for load and dump options and handle load/dump operations:
    std::string infile;
    infile = parser.get<std::string>("load");
    if (infile != "") {
        std::ifstream in(infile.c_str(), std::ios_base::in);
        if (!parser.load(in)) {
            std::cerr << "Can't open or properly read a given YAML-config file." << std::endl;
            return PARSE_FATAL_FAILURE;
        }
    }
    std::string outfile;
    outfile = parser.get<std::string>("dump");
    if (outfile != "") {
        std::string out;
        out = parser.dump();
        std::ofstream of(outfile.c_str(), std::ios_base::out);
        of << out;
    }

    for (auto const &s : solver_types) {
        if (parser.is_option_defaulted(s + "_params")) {
            if (s == "solver") {
                XAMG::params::param_list list;
                list.add_value<std::string>("method", "BiCGStab");
                params.add(s, list);
            }
        } else {
            std::map<std::string, std::string> cmdline_params;
            parser.get(s + "_params", cmdline_params);
            params.add_map(s, cmdline_params);
        }
        if (std::regex_search(s, std::regex("smooth")) || s == "solver" || s == "preconditioner") {
            std::map<std::string, std::string> override_params;
            std::string opt = s + "_override";
            parser.get(opt, override_params);
            params.add_override(s, override_params);
        }
    }
    return PARSE_OK;
}
