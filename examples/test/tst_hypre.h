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

#include <hypre/hypre_wrapper.h>

void tst_hypre_test(XAMG::matrix::matrix &m, XAMG::vector::vector &x, XAMG::vector::vector &y,
                    const XAMG::params::global_param_list &params) {

    for (int it = 0; it < 3; ++it) {
        XAMG::blas::set_const<float64_t, NV>(x, 0, true);
        XAMG::out.norm<float64_t, NV>(x, " x0 ");
        XAMG::out.norm<float64_t, NV>(y, " y0 ");

        if (NV == 1) {
            XAMG::hypre::solve(m, x, y, params);

            XAMG::out.norm<float64_t, NV>(x, " x ");
            //            XAMG::XAMG::out.norm<float64_t, NV>(y, " YY ");
        } else {
            XAMG::out << XAMG::WARN << "Hypre test mode is allowed for single rhs vector only\n";
        }
    }
}
