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

#include "io/io.h"
#include "param/params.h"

namespace XAMG {
namespace hypre {

template <typename F>
void get_full_hierarchy(matrix::matrix &m, std::vector<matrix::mg_layer> &mg_tree,
                        const params::global_param_list &global_list);
template <typename F>
void get_per_level_hierarchy(matrix::matrix &m, std::vector<matrix::mg_layer> &mg_tree,
                             const params::global_param_list &global_list);

void solve(matrix::matrix &m, vector::vector &x, vector::vector &y,
           const params::global_param_list &params);

} // namespace hypre
} // namespace XAMG

#ifndef XAMG_SEPARATE_OBJECT
#include "hypre_wrapper_inline.h"
#endif
