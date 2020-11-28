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

template <typename F>
void inverse(const dense_matrix<F> &mat, dense_matrix<F> &mat_inv) {
    assert(mat.nrows == mat.ncols);

    dense_matrix<F> mat_ext(mem::CORE);
    mat_ext.nrows = mat.nrows;
    mat_ext.ncols = 2 * mat.ncols;
    mat_ext.alloc();
    blas::set_const<F, 1>(mat_ext.val, 0.0, true);

    mat.val.check(vector::vector::initialized);
    auto val_ptr = mat.val.template get_aligned_ptr<F>();
    auto val_ext_ptr = mat_ext.val.template get_aligned_ptr<F>();

    for (uint64_t i = 0; i < mat_ext.nrows; ++i) {
        val_ext_ptr[i * mat_ext.ncols + i + mat.ncols] = 1.0;

        for (uint64_t j = 0; j < mat.ncols; ++j)
            val_ext_ptr[i * mat_ext.ncols + j] = val_ptr[i * mat.ncols + j];
    }
    mat_ext.val.if_initialized = true;
    mat_ext.val.if_zero = false;

    ///////////////////
    // Inversion

    for (uint64_t i = 0; i < mat_ext.nrows; ++i) {
        auto val_ptr_i = val_ext_ptr + i * mat_ext.ncols;
        for (uint64_t k = mat_ext.ncols - 1; k > i; --k)
            val_ptr_i[k] /= val_ptr_i[i];
        val_ptr_i[i] = 1.0;

        for (uint64_t j = i + 1; j < mat_ext.nrows; ++j) {
            auto val_ptr_j = val_ext_ptr + j * mat_ext.ncols;
            for (uint64_t k = mat_ext.ncols - 1; k > i; --k)
                val_ptr_j[k] -= val_ptr_j[i] * val_ptr_i[k];
            val_ptr_j[i] = 0.0;
        }
    }

    for (uint64_t i = 0; i < mat_ext.nrows; ++i) {
        uint64_t row = mat_ext.nrows - 1 - i;
        auto val_ptr_i = val_ext_ptr + row * mat_ext.ncols;

        for (uint64_t j = 0; j < row; ++j) {
            auto val_ptr_j = val_ext_ptr + j * mat_ext.ncols;
            for (uint64_t k = mat_ext.ncols / 2; k < mat_ext.ncols; ++k)
                val_ptr_j[k] -= val_ptr_j[row] * val_ptr_i[k];
            val_ptr_j[row] = 0.0;
        }
    }
    //    mat_ext.print();

    /////////

    mat_inv.nrows = mat_ext.nrows;
    mat_inv.ncols = mat_ext.ncols / 2;
    mat_inv.block_nrows = mat_inv.nrows;
    mat_inv.block_ncols = mat_inv.ncols;
    mat_inv.alloc();
    // mat_inv.val.check(vector::vector::allocated);
    auto val_inv_ptr = mat_inv.val.template get_aligned_ptr<F>();

    for (uint64_t i = 0; i < mat_inv.nrows; ++i) {
        uint64_t ii = i + mat_inv.block_row_offset;
        auto val_ext_ptr_i = val_ext_ptr + ii * mat_ext.ncols + mat_ext.ncols / 2;
        auto val_inv_ptr_i = val_inv_ptr + i * mat_inv.ncols;

        for (uint64_t j = 0; j < mat_inv.ncols; ++j)
            val_inv_ptr_i[j] = val_ext_ptr_i[j];
    }

    mat_inv.val.if_initialized = true;
    mat_inv.val.if_zero = false;
    // mat_inv.print();
}

template <typename F>
void inverse(const matrix &A, matrix &A_inv) {
    dense_matrix<F> mat_row(mem::CORE);
    dense_matrix<F> mat_node(mem::CORE);
    dense_matrix<F> mat_global(mem::CORE);
    auto &numa_layer = A.data_layer.find(segment::NUMA)->second;
    auto &node_layer = A.data_layer.find(segment::NODE)->second;
    A_inv.reduced_prec = A.reduced_prec;

    if (id.numa_master_process()) {
        std::vector<dense_matrix<F>> blocks_in;
        blocks_in.push_back(numa_layer.diag.data);

        for (uint32_t np = 0; np < numa_layer.offd.size(); ++np)
            blocks_in.push_back(numa_layer.offd[np].data);

        for (uint32_t np = 0; np < node_layer.offd.size(); ++np)
            blocks_in.push_back(node_layer.offd[np].data);

        merge_row_blocks(blocks_in, mat_row);
    }

    if (id.numa_master_process()) {
        collect(mat_row, mat_node, mpi::CROSS_NUMA);
    }

    if (id.node_master_process()) {
        collect(mat_node, mat_global, mpi::CROSS_NODE);
        // if (!id.gl_proc)
        //    mat_global.print();
    }

    ////

    A_inv.segmentation = SEGMENT_BY_SLICES;
    A_inv.set_part(A.row_part);
    A_inv.add_data_layer(segment::NUMA);
    auto &inv_numa_layer = A_inv.data_layer.find(segment::NUMA)->second;

    dense_matrix<F> mat_inv(mem::CORE);
    if (id.master_process())
        inverse<F>(mat_global, mat_inv);

    dense_matrix<F> node_inv(mem::CORE);
    if (id.node_master_process())
        distribute(mat_inv, node_inv, A_inv.row_part->node_layer, mpi::CROSS_NODE);

    dense_matrix<F> numa_inv(mem::CORE);
    if (id.numa_master_process()) {
        distribute(node_inv, numa_inv, A_inv.row_part->numa_layer, mpi::CROSS_NUMA);

        A_inv.min_eig = std::numeric_limits<float64_t>::signaling_NaN();
        A_inv.max_eig = std::numeric_limits<float64_t>::signaling_NaN();
        inv_numa_layer.diag.reduced_prec = A_inv.reduced_prec;
        inv_numa_layer.diag.assemble(numa_inv, id.nd_numa);
    } else {
        // Dummy matrix assembling to avoid problems with blas2_drivers allocation
        dense_matrix<F> dummy(mem::CORE);
        inv_numa_layer.diag.assemble(dummy, id.nd_numa);
    }
}

} // namespace matrix
} // namespace XAMG
