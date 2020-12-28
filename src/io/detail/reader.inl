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
namespace io {

template <typename T>
bool read_vector(vector::vector &vec, const uint64_t offset, FILE *file) {
    vec.check(vector::vector::allocated);

    auto vec_ptr = vec.get_aligned_ptr<T>();

    fseek(file, offset * sizeof(T), SEEK_CUR);
    if (fread(vec_ptr, sizeof(T), vec.size * vec.nv, file) != vec.size * vec.nv)
        return false;
    else {
        vec.if_initialized = true;
        vec.if_zero = false;
        return true;
    }
}

template <class MATRIX_TYPE, typename FP, typename ROW_IDX_TYPE, typename COL_IDX_TYPE>
bool read_matrix(MATRIX_TYPE &mat, FILE *file) {

    uint32_t i32_nrows, i32_nonzeros;

    fread(&i32_nrows, sizeof(uint32_t), 1, file);
    fread(&i32_nonzeros, sizeof(uint32_t), 1, file);

    // XAMG::out << "Global: "<< i32_nrows << " || " << i32_nonzeros << "\n";

    //////////

    uint64_t block_size = i32_nrows / id.gl_nprocs;
    uint64_t block_offset = block_size * id.gl_proc;
    if (id.gl_proc == id.gl_nprocs - 1)
        block_size = i32_nrows - block_offset;

    uint32_t i1, i2;

    uint64_t file_offset = 2 * sizeof(uint32_t) + block_offset * sizeof(uint32_t);
    fseek(file, file_offset, SEEK_SET);
    fread(&i1, sizeof(int), 1, file);

    file_offset += block_size * sizeof(int);
    fseek(file, file_offset, SEEK_SET);
    fread(&i2, sizeof(int), 1, file);

    uint64_t block_nonzeros = i2 - i1;

    //////////

    mat.nrows = block_size;
    mat.block_nrows = block_size;
    mat.ncols = i32_nrows;
    mat.block_ncols = i32_nrows;
    mat.block_row_offset = block_offset;
    mat.block_col_offset = 0;
    mat.nonzeros = block_nonzeros;

    mat.alloc();
    XAMG::out << XAMG::DBG << "Local: " << mat.nrows << " || " << mat.nonzeros << "\n";

    /////////

    if (mat.if_empty)
        return true;
    uint64_t row_offset = mat.block_row_offset;
    file_offset = 2 * sizeof(uint32_t);
    fseek(file, file_offset, SEEK_SET);

    if (!read_vector<ROW_IDX_TYPE>(mat.row, row_offset, file))
        return false;
    file_offset += (i32_nrows + 1) * sizeof(uint32_t);

    mat.row.check(vector::vector::initialized);
    auto row_ptr = mat.row.template get_aligned_ptr<ROW_IDX_TYPE>();
    uint64_t nnz_offset = row_ptr[0];

    fseek(file, file_offset, SEEK_SET);
    if (!read_vector<COL_IDX_TYPE>(mat.col, nnz_offset, file))
        return false;
    file_offset += i32_nonzeros * sizeof(uint32_t);

    fseek(file, file_offset, SEEK_SET);
    if (!read_vector<FP>(mat.val, nnz_offset, file))
        return false;
    file_offset += i32_nonzeros * sizeof(float64_t);

    fseek(file, file_offset, SEEK_SET);

    /////////

    for (uint64_t i = 0; i < mat.row.size; ++i)
        row_ptr[i] -= nnz_offset;

    return true;
}

template <class MATRIX_TYPE, uint16_t NV>
bool read_system(MATRIX_TYPE &mat, vector::vector &x, vector::vector &b, const std::string &path) {
    using FP = typename MATRIX_TYPE::float_type;
    using ROW_IDX_TYPE = typename MATRIX_TYPE::row_idx_type;
    using COL_IDX_TYPE = typename MATRIX_TYPE::col_idx_type;

#ifdef XAMG_DBG_HEADER
    XAMG::out << FUNC_PREFIX << "CSR read data, fp: <" << DEMANGLE_TYPEID_NAME(FP) << ">; I1: <"
              << DEMANGLE_TYPEID_NAME(ROW_IDX_TYPE) << ">; I2: <"
              << DEMANGLE_TYPEID_NAME(COL_IDX_TYPE) << ">\n";
#endif

    //  parallel data reader

    FILE *file = fopen(path.c_str(), "rb");
    assert(file != NULL);

    bool flag = read_matrix<MATRIX_TYPE, FP, ROW_IDX_TYPE, COL_IDX_TYPE>(mat, file);
    assert(flag);

    //////////
    //  square matrix expected...
    uint32_t i32_nrows = mat.ncols;
    uint64_t row_offset = mat.block_row_offset * NV;

    x.alloc<FP>(mat.nrows, NV);
    b.alloc<FP>(mat.nrows, NV);

    if (read_vector<FP>(b, row_offset, file)) {
        XAMG::out << "B vector read completed...\n";

        uint64_t file_offset = (i32_nrows * NV - row_offset - b.size * NV) * sizeof(float64_t);
        fseek(file, file_offset, SEEK_SET);

        if (read_vector<FP>(x, row_offset, file)) {
            XAMG::out << "X vector read completed...\n";
        } else {
            XAMG::out << "X vector read failed...\n";
            blas::set_const<FP, NV>(x, 0.0, true);
        }
    } else {
        XAMG::out << "B vector read failed...\n";
        blas::set_const<FP, NV>(b, 1.0, true);
    }

    fclose(file);

    //////////

    XAMG::out << "Data read completed\n";

    return true;
}

} // namespace io
} // namespace XAMG
