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

namespace matrix {

template <typename F, typename I>
void XAMG::matrix::matrix::segment(ell_matrix<F, I> &mat_ell, const uint32_t nblocks) {
    //    assert(0);
    assert(nblocks == 1);

    std::vector<uint64_t> &row_block_size = row_part->node_layer.block_size;
    std::vector<uint64_t> &row_block_indx = row_part->node_layer.block_indx;

    std::vector<uint64_t> &col_block_size = col_part->node_layer.block_size;
    std::vector<uint64_t> &col_block_indx = col_part->node_layer.block_indx;
    assert(col_block_size.size() == nblocks);

    std::vector<ell_matrix<F, I>> blocks; //(nblocks);
    for (uint16_t nb = 0; nb < nblocks; ++nb)
        blocks.emplace_back(ell_matrix<F, I>(mem::LOCAL));
    blocks[0] = mat_ell;

    /////////

    //    ...

    /////////
    std::vector<uint64_t> block_mapping_uncompress(1, id.gl_proc);
    assemble(blocks, block_mapping_uncompress);

    /////////

    collect_matrix_stats();
}

template <typename F, typename I>
void XAMG::matrix::matrix::assemble(const std::vector<ell_matrix<F, I>> &blocks,
                                    const std::vector<uint64_t> &block_mapping_uncompress) {
    min_eig = std::numeric_limits<float64_t>::signaling_NaN();
    max_eig = std::numeric_limits<float64_t>::signaling_NaN();
    auto &numa_layer = data_layer.find(segment::NUMA)->second;
    auto &node_layer = data_layer.find(segment::NODE)->second;

    assert(blocks.size() == block_mapping_uncompress.size());

    uint32_t block_cntr = 0;
    uint32_t non_empty_blocks = blocks.size();

    node_layer.offd.resize(non_empty_blocks - 1);
    //    assert(0);

    for (uint32_t nb = 0; nb < non_empty_blocks; ++nb) {
        //        uint16_t block_hash = bit_encoding<F>(); // | encode_ell_hash(blocks[nb].nrows);
        uint16_t block_hash = bit_encoding<F>(); // | encode_csr_hash(blocks[nb].nonzeros, ncols,
                                                 // blocks[nb].nrows, blocks[nb].ncols);
                                                 //        print_bits(block_hash);

        std::shared_ptr<backend> backend_obj =
            create_matrix_block_obj(blocks[nb], block_hash, mem::LOCAL);

        uint32_t rnb = block_mapping_uncompress[nb];
        if (rnb == (uint32_t)id.gl_proc) {
            numa_layer.diag.data = std::move(backend_obj);
            numa_layer.diag.hash = block_hash;
            numa_layer.diag.block_id = rnb;
        } else {
            node_layer.offd[block_cntr].data = std::move(backend_obj);
            node_layer.offd[block_cntr].hash = block_hash;
            node_layer.offd[block_cntr].block_id = rnb;
            ++block_cntr;
        }
    }
}

} // namespace matrix
} // namespace XAMG
