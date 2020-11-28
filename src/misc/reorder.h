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

#include <vector>
#include "xamg_types.h"
#include "comm/mpi_wrapper.h"

namespace XAMG {

struct map_info {
    // const std::vector<int> &permutation;        // initial vector generated, e.g., by graph partitioning methods; reflects each element to new block
    std::vector<int>
        list_of_neighbours; // local list of neighbour procs the data must be redistributed to
    std::vector<uint32_t> nrows_per_block; // number of rows to be sent to each neighbour proc
    std::vector<int> mapped_block;         // reflection of local element to new mapped block id
    uint32_t nneighbours;

    map_info(const std::vector<int> &permutation, const segment::hierarchy &layer) {
        mpi::comm_pool comm_group(mpi::cross_layer_comm(layer));

        std::vector<bool> nbr_flags(comm_group.nprocs, false);
        for (auto elem : permutation)
            nbr_flags[elem] = true;

        std::vector<int> nbr_list_mapping(comm_group.nprocs, -1);
        for (size_t i = 0; i < (size_t)comm_group.nprocs; ++i) {
            if (nbr_flags[i]) {
                nbr_list_mapping[i] = list_of_neighbours.size();
                list_of_neighbours.push_back(i);
            }
        }

        nneighbours = list_of_neighbours.size();

        mapped_block.assign(permutation.size(), 0);
        nrows_per_block.assign(nneighbours, 0);
        for (uint32_t i = 0; i < permutation.size(); ++i) {
            size_t to_block = permutation[i];
            mapped_block[i] = nbr_list_mapping[to_block];

            ++nrows_per_block[mapped_block[i]];
        }
    }
};

} // namespace XAMG
