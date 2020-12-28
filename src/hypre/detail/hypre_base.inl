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
namespace hypre {

#define XAMG_HYPRE_REPRODUCIBLE_MODE

#ifndef XAMG_SEPARATE_OBJECT
void hypre_base::set_matrix_diag_block(const matrix::matrix_block &diag, const uint32_t &offset,
                                       const uint32_t &size) {
    assert(!diag.data->if_empty);
    uint64_t row_offset = diag.data->get_block_row_offset();
    uint64_t col_offset = diag.data->get_block_col_offset();

    std::vector<int> col;
    std::vector<float64_t> val;

    // block disassemble!
    for (uint32_t l = offset; l < offset + size; ++l) {
        int row_size = diag.data->get_row_size(l);
        diag.data->get_row(l, col, val);

        int ll = l + row_offset;
        for (uint32_t ii = 0; ii < (uint32_t)row_size; ++ii)
            col[ii] += col_offset;

        HYPRE_IJMatrixSetValues(hA, 1, &row_size, &ll, col.data(), val.data());
    }
}

void hypre_base::set_matrix_offd_block(const matrix::matrix_block &offd,
                                       const std::pair<uint64_t, uint64_t> &range) {
    if (offd.data->if_empty)
        return;
    uint64_t row_offset = offd.data->get_block_row_offset();
    uint64_t col_offset = offd.data->get_block_col_offset();

    std::vector<int> col;
    std::vector<float64_t> val;

    for (uint32_t l = range.first; l < range.second; ++l) {
        int row_size = offd.data->get_row_size(l);
        offd.data->get_row(l, col, val);

        int ll = offd.data->unpack_row_indx(l) + row_offset;
        for (uint32_t i = 0; i < col.size(); ++i)
            col[i] = offd.data->unpack_col_indx(col[i]) + col_offset;

        HYPRE_IJMatrixSetValues(hA, 1, &row_size, &ll, col.data(), val.data());
    }
}

void hypre_base::assemble_matrix(const matrix::matrix &m) {
    uint64_t l;
    // Check: square input matrix only allowed
    assert(m.row_part->node_layer.block_indx.back() + m.row_part->node_layer.block_size.back() ==
           m.col_part->node_layer.block_indx.back() + m.col_part->node_layer.block_size.back());

    uint64_t core_size, core_offset;
    auto &numa_layer = m.data_layer.find(segment::NUMA)->second;
    auto &node_layer = m.data_layer.find(segment::NODE)->second;

    core_size = m.row_part->core_layer.block_size[id.nm_core];
    core_offset = m.row_part->core_layer.block_indx[id.nm_core];

    int row_block_indx = numa_layer.diag.data->block_row_offset + core_offset;
    int row_block_size = core_size;
    int col_block_indx = row_block_indx;
    int col_block_size = row_block_size;

    HYPRE_IJMatrixCreate(*((MPI_Comm *)id.get_comm()), row_block_indx,
                         row_block_indx + row_block_size - 1, col_block_indx,
                         col_block_indx + col_block_size - 1, &hA);

    HYPRE_IJMatrixSetObjectType(hA, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(hA);

    ///////////////////
    //        consistency check:
    for (uint32_t nb = 0; nb < numa_layer.offd.size(); ++nb) {
        if (numa_layer.offd[nb].data->if_empty)
            continue;
        assert(numa_layer.diag.data->get_block_row_offset() ==
               numa_layer.offd[nb].data->get_block_row_offset());
    }

    for (uint32_t nb = 0; nb < node_layer.offd.size(); ++nb) {
        if (node_layer.offd[nb].data->if_empty)
            continue;
        assert(part->node_layer.block_indx[id.gl_node] + part->numa_layer.block_indx[id.nd_numa] ==
               node_layer.offd[nb].data->get_block_row_offset());
    }

/////////
#ifndef XAMG_HYPRE_REPRODUCIBLE_MODE
    std::vector<std::pair<uint64_t, uint64_t>> numa_range =
        m.get_rows_range(core_offset, core_size, segment::NUMA);
    std::vector<std::pair<uint64_t, uint64_t>> node_range =
        m.get_rows_range(core_offset, core_size, segment::NODE);

    assert(numa_range.size() == numa_layer.offd.size());
    assert(node_range.size() == node_layer.offd.size());

    //    XAMG::out << "offset " << core_offset << " size " << core_size << std::endl;
    set_matrix_diag_block(numa_layer.diag, core_offset, core_size);

    for (uint32_t nb = 0; nb < numa_layer.offd.size(); ++nb)
        set_matrix_offd_block(numa_layer.offd[nb], numa_range[nb]);

    for (uint32_t nb = 0; nb < node_layer.offd.size(); ++nb)
        set_matrix_offd_block(node_layer.offd[nb], node_range[nb]);
#else
    std::vector<int> col;
    std::vector<float64_t> val;

    uint64_t row_offset = numa_layer.diag.data->get_block_row_offset();
    uint64_t col_offset = numa_layer.diag.data->get_block_col_offset();

    std::vector<uint32_t> numa_offd_indx(numa_layer.offd.size(), 0);
    std::vector<uint32_t> node_offd_indx(node_layer.offd.size(), 0);

    for (uint32_t l = core_offset; l < core_offset + core_size; ++l) {
        int row_size = numa_layer.diag.data->get_row_size(l);
        numa_layer.diag.data->get_row(l, col, val);

        int ll = l + row_offset;
        for (uint32_t ii = 0; ii < (uint32_t)row_size; ++ii)
            col[ii] += col_offset;

        std::vector<int> col0;
        std::vector<float64_t> val0;

        for (uint32_t nb = 0; nb < numa_layer.offd.size(); ++nb) {
            if (numa_layer.offd[nb].data->if_empty)
                continue;
            numa_layer.offd[nb].data->get_row_and_unpack(l, col0, val0, numa_offd_indx[nb]);
            for (uint32_t i = 0; i < col0.size(); ++i) {
                col.push_back(col0[i]);
                val.push_back(val0[i]);
            }
        }

        for (uint32_t nb = 0; nb < node_layer.offd.size(); ++nb) {
            if (node_layer.offd[nb].data->if_empty)
                continue;
            node_layer.offd[nb].data->get_row_and_unpack(l, col0, val0, node_offd_indx[nb]);
            for (uint32_t i = 0; i < col0.size(); ++i) {
                col.push_back(col0[i]);
                val.push_back(val0[i]);
            }
        }

        // sort();
        for (uint32_t iii = 0; iii < col.size(); ++iii) {
            for (uint32_t jjj = iii + 1; jjj < col.size(); ++jjj) {
                if (col[iii] > col[jjj]) {
                    std::swap(col[iii], col[jjj]);
                    std::swap(val[iii], val[jjj]);
                }
            }
        }

        //        for (int iii = 0; iii < col.size(); ++iii)
        //            XAMG::out << ALLRANKS << "row "<< ll << " " << col[iii] << std::endl;

        row_size = col.size();
        HYPRE_IJMatrixSetValues(hA, 1, &row_size, &ll, col.data(), val.data());
    }
#endif

    HYPRE_IJMatrixAssemble(hA);
    //    HYPRE_IJMatrixPrint(hA, "mat.txt");
}

void hypre_base::assemble_vector(HYPRE_IJVector &hb, const vector::vector &b) {
    b.check(vector::vector::initialized);
    uint64_t row_block_size = part->core_layer.block_size[id.nm_core];
    uint64_t row_block_indx = part->core_layer.block_indx[id.nm_core];

    HYPRE_IJVectorCreate(*((MPI_Comm *)id.get_comm()), row_block_indx,
                         row_block_indx + row_block_size - 1, &hb);
    HYPRE_IJVectorSetObjectType(hb, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(hb);

    ///////////////////

    if (row_indx.size() == 0) {
        row_indx.resize(row_block_size);

        for (uint64_t l = 0; l < (uint64_t)row_block_size; ++l)
            row_indx[l] = l + row_block_indx;
    } else {
        assert(row_indx.size() == row_block_size);
    }

    auto b_ptr = b.get_aligned_ptr<float64_t>();
    HYPRE_IJVectorSetValues(hb, row_block_size, row_indx.data(), b_ptr);
    HYPRE_IJVectorAssemble(hb);
}

void hypre_base::get_vector_data(HYPRE_IJVector &hb, vector::vector &b) {
    b.check(vector::vector::allocated);
    uint64_t row_block_size = part->core_layer.block_size[id.nm_core];
    uint64_t row_block_indx = part->core_layer.block_indx[id.nm_core];
    assert(row_indx.size() == row_block_size);

    auto b_ptr = b.get_aligned_ptr<float64_t>();
    HYPRE_IJVectorGetValues(hb, row_block_size, row_indx.data(), b_ptr);
    b.if_initialized = true;
    b.if_zero = false;
}

template <typename F>
void hypre_base::parse_matrix(
    matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t> &mat_csr,
    const hypre_ParCSRMatrix *hypre_matrix) {
    hypre_CSRMatrix *mat_diag, *mat_offd;
    uint64_t nrows, ncols;

    mat_diag = hypre_ParCSRMatrixDiag(hypre_matrix);
    mat_offd = hypre_ParCSRMatrixOffd(hypre_matrix);

    HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(hypre_matrix);
    HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(hypre_matrix);
    HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(hypre_matrix);

    uint64_t block_nrows = mat_diag->num_rows;
    uint64_t block_ncols = mat_diag->num_cols;

    nrows = mat_diag->num_rows;
    ncols = hypre_matrix->global_num_cols;
    uint64_t nonzeros = mat_diag->num_nonzeros + mat_offd->num_nonzeros;

    mat_csr.nrows = nrows;
    mat_csr.nonzeros = nonzeros;
    mat_csr.ncols = ncols;
    mat_csr.block_nrows = nrows;
    mat_csr.block_ncols = ncols;
    mat_csr.block_row_offset = first_row_index;
    mat_csr.block_col_offset = 0;

    mat_csr.alloc();

    if (mat_csr.if_empty)
        return;
    /////////

    uint32_t cntr = 0;
    std::vector<uint64_t> col;
    std::vector<F> val;
    uint32_t row_size = 0;

    for (uint64_t l = 0; l < nrows; ++l)
        row_size = std::max(row_size, (uint32_t)(mat_diag->i[l + 1] - mat_diag->i[l] +
                                                 mat_offd->i[l + 1] - mat_offd->i[l]));

    col.resize(row_size);
    val.resize(row_size);

    mat_csr.upload_row(0, col, val, 0);

    for (uint64_t l = 0; l < nrows; ++l) {
        cntr = 0;

        for (uint64_t k = mat_diag->i[l]; k < (uint64_t)mat_diag->i[l + 1]; ++k) {
            val[cntr] = (F)mat_diag->data[k];
            col[cntr] = mat_diag->j[k] + first_col_diag;
            cntr++;
        }

        for (uint64_t k = mat_offd->i[l]; k < (uint64_t)mat_offd->i[l + 1]; ++k) {
            val[cntr] = (F)mat_offd->data[k];
            col[cntr] = col_map_offd[mat_offd->j[k]];
            cntr++;
        }

        mat_csr.upload_row(l, col, val, cntr);
    }

    assert(mat_csr.row.template get_aligned_ptr<uint32_t>()[mat_csr.get_nrows()] ==
           mat_csr.get_nonzeros());
}

void hypre_base::get_part(part::part &part, const hypre_ParCSRMatrix *hypre_matrix) {
    hypre_CSRMatrix *mat_diag = hypre_ParCSRMatrixDiag(hypre_matrix);
    assert(mat_diag->num_rows == mat_diag->num_cols);

    uint64_t block_size = mat_diag->num_rows;
    part.get_part(block_size);
}

void hypre_base::create_hierarchy(const params::global_param_list &global_list) {
    double t1, t2;
    create_multigrid_solver(solver, global_list);

    mpi::barrier(mpi::GLOBAL);
    t1 = sys::timer();

    setup_multigrid_solver();

    mpi::barrier(mpi::GLOBAL);
    t2 = sys::timer();

    XAMG::out << "AMG setup time = " << t2 - t1 << " sec" << std::endl;
}

template <typename F>
void hypre_base::parse_hierarchy(std::vector<matrix::mg_layer> &mg_tree, const bool reduced_prec) {
    hypre_ParAMGData *amg_data = (hypre_ParAMGData *)solver;
    hypre_ParCSRMatrix **A_array = amg_data->A_array;
    hypre_ParCSRMatrix **P_array = amg_data->P_array;

    assert(mg_tree.size() > 0);
    size_t start_level = mg_tree.size();
    size_t num_levels = start_level + amg_data->num_levels - 1;

    const mem::allocation alloc_mode = mg_tree[0].get_A().alloc_mode;
    mg_tree.resize(num_levels, matrix::mg_layer(alloc_mode));

    // Parsing A-hierarchy
    for (uint16_t nl = start_level; nl < mg_tree.size(); ++nl) {
        mg_tree[nl].part = part::get_shared_part();
        get_part(*(mg_tree[nl].part), A_array[nl - (start_level - 1)]);
    }

    for (uint16_t nl = start_level; nl < mg_tree.size(); ++nl) {
        matrix::matrix &A = mg_tree[nl].get_A();
        A.set_part(mg_tree[nl].part);

        matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t> mat_csr;
        parse_matrix(mat_csr, A_array[nl - (start_level - 1)]);
        A.reduced_prec = reduced_prec;
        A.construct(mat_csr);
    }

    for (uint16_t nl = start_level - 1; nl < mg_tree.size(); ++nl) {
        matrix::matrix &A = mg_tree[nl].get_A();
        if (amg_data->min_eig_est != nullptr)
            A.min_eig = amg_data->min_eig_est[nl - (start_level - 1)];
        if (amg_data->max_eig_est != nullptr)
            A.max_eig = amg_data->max_eig_est[nl - (start_level - 1)];
    }

    /////////
    // Parsing R/P-hierarchy
    for (uint16_t nl = start_level - 1; nl < mg_tree.size() - 1; ++nl) {
        matrix::matrix &P = mg_tree[nl].get_P();
        matrix::matrix &R = mg_tree[nl].get_R();
        P.set_part(mg_tree[nl].part, mg_tree[nl + 1].part);
        R.set_part(mg_tree[nl + 1].part, mg_tree[nl].part);

        matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t> mat_csr;
        parse_matrix(mat_csr, P_array[nl - (start_level - 1)]);

        matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t> node_csr;
        matrix::collect(mat_csr, node_csr, mpi::INTRA_NODE);

        std::vector<matrix::csr_mtx_chunk_pair<F, uint32_t, uint32_t, uint32_t, uint32_t>>
            csr_node_chunks;
        std::vector<matrix::csr_mtx_chunk_pair<F, uint32_t, uint32_t, uint32_t, uint32_t>>
            csr_nodeT_chunks;

        if (id.node_master_process()) {
            split_by_columns(node_csr, csr_node_chunks, P.col_part->node_layer);
            matrix::compress_chunks(csr_node_chunks, segment::NODE);
        }
        P.reduced_prec = R.reduced_prec = reduced_prec;
        P.construct(csr_node_chunks);

        matrix::transpose(csr_node_chunks, csr_nodeT_chunks);
        R.construct(csr_nodeT_chunks);
    }
}
#endif

} // namespace hypre
} // namespace XAMG
