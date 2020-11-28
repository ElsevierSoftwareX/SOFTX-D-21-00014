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

template <typename F>
void get_full_hierarchy(matrix::matrix &m, std::vector<matrix::mg_layer> &mg_tree,
                        const params::global_param_list &global_list) {
    assert(m.row_part == m.col_part);
    const auto &list = global_list.get_if<std::string>({{"method", "MultiGrid"}});
    bool mg_reduced_precision = list.get_value<uint16_t>("mg_reduced_precision");

    mg_tree.resize(1, matrix::mg_layer(m.alloc_mode));
    mg_tree[0].A.set_ref(m);
    mg_tree[0].part = m.row_part;

    hypre_base hypre_obj(mg_tree[0].part);

    hypre_obj.assemble_matrix(mg_tree[0].get_A());
    hypre_obj.get_matrix_object();

    hypre_obj.create_hierarchy(global_list);
    hypre_obj.parse_hierarchy<F>(mg_tree, mg_reduced_precision);

    hypre_obj.destroy_multigrid_solver();
    hypre_obj.destroy_matrix();
}

template <typename F>
void get_per_level_hierarchy(matrix::matrix &m, std::vector<matrix::mg_layer> &mg_tree,
                             const params::global_param_list &global_list) {
    assert(m.row_part == m.col_part);

    mg_tree.resize(1, matrix::mg_layer(m.alloc_mode));
    mg_tree[0].A.set_ref(m);
    mg_tree[0].part = m.row_part;

    std::string solver_type;
    const auto &list = global_list.get_if<std::string>({{"method", "MultiGrid"}}, solver_type);
    uint16_t mg_max_levels = list.get_value<uint16_t>("mg_max_levels");
    uint16_t mg_coarse_matrix_size = list.get_value<uint16_t>("mg_coarse_matrix_size");
    uint16_t mg_agg_num_levels = list.get_value<uint16_t>("mg_agg_num_levels");
    bool mg_reduced_precision = list.get_value<uint16_t>("mg_reduced_precision");

    size_t lev = 0;
    bool completed = false;
    while (!completed) {
        uint16_t agg_lev = 0;
        if (mg_agg_num_levels > lev)
            agg_lev = mg_agg_num_levels - lev;
        params::global_param_list modified_list(global_list);
        modified_list.forced_change_value<uint16_t>(solver_type, "mg_max_levels", 2);
        modified_list.forced_change_value<uint16_t>(solver_type, "mg_agg_num_levels", agg_lev);

        auto solver_local_list = modified_list.get("solver", lev);
        auto preconditioner_local_list = modified_list.get("preconditioner", lev);
        auto pre_smoother_local_list = modified_list.get("pre_smoother", lev);
        auto post_smoother_local_list = modified_list.get("post_smoother", lev);
        modified_list.get("solver").override_params(solver_local_list);
        modified_list.get("preconditioner").override_params(preconditioner_local_list);
        modified_list.get("pre_smoother").override_params(pre_smoother_local_list);
        modified_list.get("post_smoother").override_params(post_smoother_local_list);

        hypre_base hypre_obj(mg_tree[lev].part);

        hypre_obj.assemble_matrix(mg_tree[lev].get_A());
        hypre_obj.get_matrix_object();

        hypre_obj.create_hierarchy(modified_list);
        hypre_obj.parse_hierarchy<F>(mg_tree, mg_reduced_precision);

        hypre_obj.destroy_multigrid_solver();
        hypre_obj.destroy_matrix();

        // XAMG::out << "level : " << lev << std::endl;
        ++lev;

        size_t mat_size = mg_tree.back().get_A().col_part->node_layer.block_indx[id.gl_nnodes];
        if ((lev == mg_max_levels) || (mg_coarse_matrix_size >= mat_size)) {
            completed = true;
        }
    }
}

} // namespace hypre
} // namespace XAMG
