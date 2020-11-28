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

#include "xamg_types.h"

void graph_decomp(int *iproc_indx, int *row_ptr, int *col_ptr, int *vwgt, int nprocs,
                  int *partitioning, const XAMG::segment::hierarchy layer) {
    int options[5];
    int wgtflag = 1;
    int numflag = 0;
    int vol = 0;
    int ncon = 1;
    float ubvec = 1.01;
    std::vector<float> tpwgt(nprocs, (1. / nprocs));
    MPI_Comm *mpi_comm = (MPI_Comm *)id.get_comm(layer);

    ParMETIS_V3_PartKway(iproc_indx, row_ptr, col_ptr, vwgt, NULL, &wgtflag, &numflag, &ncon,
                         &nprocs, tpwgt.data(), &ubvec, options, &vol, partitioning, mpi_comm);
}
