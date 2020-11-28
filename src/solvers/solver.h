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

#include "xamg_headers.h"
#include "xamg_types.h"

#include "primitives/matrix/matrix.h"
#include "primitives/vector/vector.h"

#include "blas/blas.h"
#include "blas/blas_merged.h"

#include "param/params.h"

#include "hypre/hypre_wrapper.h"

#include "comm/shm_allreduce.h"

namespace XAMG {
namespace solver {

/////////

struct solver_stats {
    std::vector<bool> if_converged;
    std::vector<float64_t> abs_res;
    std::vector<float64_t> rel_res;
    uint32_t iters;
    solver_stats() : if_converged(0), abs_res(0), rel_res(0), iters(0) {}
    void reset(const uint16_t conv_check) {
        assert(if_converged.size() == abs_res.size());
        assert(abs_res.size() == rel_res.size());

        iters = 0;
        if (conv_check) {
            for (size_t i = 0; i < if_converged.size(); ++i) {
                if_converged[i] = false;
                abs_res[i] = rel_res[i] = 0.0;
            }
        }
    }
};

class base_solver_interface;
template <typename F, uint16_t NV>
class base_solver;
template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> make_solver(const std::string &method, matrix::matrix &_A,
                                XAMG::vector::vector &x, XAMG::vector::vector &y);
template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> make_solver(const std::string &method, matrix::matrix &_A);

template <typename F, uint16_t NV>
std::shared_ptr<base_solver_interface>
construct_basic_solver(const params::global_param_list &params, const params::param_list &list,
                       matrix::matrix &A);

template <typename F, uint16_t NV>
std::shared_ptr<base_solver_interface>
construct_basic_solver(const params::global_param_list &params, const params::param_list &list,
                       matrix::matrix &A, XAMG::vector::vector &x, XAMG::vector::vector &y);

template <typename F, uint16_t NV>
std::shared_ptr<base_solver<F, NV>>
construct_solver_hierarchy(const params::global_param_list &params, matrix::matrix &A);

template <typename F, uint16_t NV>
std::shared_ptr<base_solver<F, NV>>
construct_solver_hierarchy(const params::global_param_list &params, matrix::matrix &A,
                           XAMG::vector::vector &x, XAMG::vector::vector &y);

struct base_solver_interface {
    params::param_list param_list;
    virtual void init() = 0;
    virtual void setup(const XAMG::params::global_param_list &params) = 0;
    //    conv vector is used to control update of solution vector for the converged RHS:
    //    in case convergence for specific RHS was achieved the overall solution typically continues
    //    as usual, but the resulting solution vector X is NOT updated, thus the displayed residual
    //    may differ from the real one
    virtual void solve(XAMG::mpi::token &tok = XAMG::mpi::null_token) = 0;
    virtual void solve(const vector::vector &conv,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) = 0;
    virtual void solve(vector::vector &_x, vector::vector &_b,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) = 0;
    virtual void solve(vector::vector &_x, vector::vector &_b, const vector::vector &conv,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) = 0;
    virtual void matrix_info() = 0;
    void renew_param_list(const params::param_list &l) { param_list.override_params(l); }
    void set_param_list(const params::param_list &l) {
        param_list.erase();
        param_list = l;
    }
    virtual void renew_params(const params::global_param_list &params, bool solver_mode = true) = 0;
    virtual ~base_solver_interface() {}
};

template <typename F, uint16_t NV>
struct base_solver : public base_solver_interface {
    using base_solver_interface::param_list;
    bool initialized = false;
    bool setup_done = false;
    matrix::matrix &A;
    vector::vector *x = nullptr;
    vector::vector *b = nullptr;
    std::vector<vector::vector> buffer;
    std::shared_ptr<base_solver_interface> precond;
    solver_stats stats;
    uint16_t buffer_nvecs = 0;
    uint16_t allreduce_buffer_size = 0;
    comm::allreduce<F, NV> allreduce_buffer;
    base_solver(matrix::matrix &_A, vector::vector &_x, vector::vector &_b)
        : A(_A), x(&_x), b(&_b) {}
    base_solver(matrix::matrix &_A) : A(_A) {}
    virtual ~base_solver() {}
    base_solver(const base_solver<F, NV> &that) = delete;
    base_solver &operator=(const base_solver<F, NV> &that) = delete;

  protected:
    void override_params(const params::param_list &l) {
        assert(!initialized);
        assert(!setup_done);
        param_list.override_params(l);
    }

  protected:
    virtual void init() { init_base(); }

    void init_base() {
        assert(!setup_done);
        auto &numa_layer = A.data_layer.find(segment::NUMA)->second;
        buffer.resize(buffer_nvecs, vector::vector(mem::DISTRIBUTED));
        for (uint16_t i = 0; i < buffer_nvecs; ++i) {
            buffer[i].alloc<F>(numa_layer.diag.data->get_nrows(), NV);
            buffer[i].set_part(A.row_part);
        }
        allreduce_buffer.alloc(allreduce_buffer_size);
        stats.abs_res.resize(NV);
        stats.rel_res.resize(NV);
        stats.if_converged.resize(NV);
        initialized = true;
    }

    virtual void setup(const XAMG::params::global_param_list &params) {
        assert(!setup_done);
        setup_done = true;
    }

  public:
    bool convergence_check(const vector::vector &res, const vector::vector &res0,
                           const vector::vector &conv) {
        assert(initialized);
        assert(setup_done);
        float32_t abs_tol, rel_tol;
        bool convergence_flag = true;
        monitor.params.start();
        param_list.get_value("abs_tolerance", abs_tol);
        param_list.get_value("rel_tolerance", rel_tol);
        monitor.params.stop();

        std::vector<F> res_elem = res.get_element<F>(0);
        std::vector<F> res0_elem = res0.get_element<F>(0);
        std::vector<F> conv_elem = conv.get_element<F>(0);
        for (uint16_t nv = 0; nv < res.nv; ++nv) {
            if (conv_elem[nv]) {
                F abs_res = sqrt(res_elem[nv]);
                F rel_res = abs_res / sqrt(res0_elem[nv]);
                if ((abs_res > abs_tol) && (rel_res > rel_tol)) {
                    convergence_flag = false;
                } else {
                    if (!stats.if_converged[nv]) {
                        stats.abs_res[nv] = abs_res;
                        stats.rel_res[nv] = rel_res;
                        stats.if_converged[nv] = true;
                    }
                }
            }
        }
        return convergence_flag;
    }

    bool converged(const vector::vector &res, const vector::vector &res0, vector::vector &conv) {
        uint16_t conv_check, conv_info;
        monitor.params.start();
        param_list.get_value("convergence_check", conv_check);
        param_list.get_value("convergence_info", conv_info);
        monitor.params.stop();

        if (conv_check) {
            if (conv_info) {
                if (!stats.iters)
                    io::print_residuals_header(0, NV);
                io::print_residuals<F>(stats.iters, res, res0, conv);
            }

            if (convergence_check(res, res0, conv)) {
                if (conv_info)
                    io::print_residuals_footer(NV);
                return true;
            }

            convergence_status(conv);
        }
        return false;
    }

    void convergence_status(vector::vector &conv_status) const {
        assert(stats.if_converged.size() == NV);

        std::vector<F> val = conv_status.get_element<F>(0);
        for (uint32_t nv = 0; nv < NV; ++nv) {
            val[nv] *= (!stats.if_converged[nv]);
        }
        conv_status.set_element<F>(0, val);

        conv_status.if_initialized = true;
        conv_status.if_zero = false;
    }

    void get_residual(const vector::vector &x0, vector::vector &r, vector::vector &res,
                      const uint16_t &conv_check = 1) {
        assert(initialized);
        assert(setup_done);
        vector::vector &b = *this->b;
        // Computes r = b - A*x
        blas2::Ax_y<F, NV>(A, x0, r, NV);
        blas::axpby<F, NV>(1.0, b, -1.0, r);

        if (conv_check) {
            uint16_t norm_type = L2_norm;
            blas::vector_norm<F, NV>(r, norm_type, res);
        }
    }

  public:
    void assemble(const params::global_param_list &params) {
        assert(!setup_done);
        assert(params.find("solver"));
        set_param_list(params.get("solver"));
        init();
        assert(initialized);
        setup(params);
        assert(setup_done);
        if (params.find("preconditioner")) {
            auto &prms = params.get("preconditioner");
            auto method = prms.get_value<std::string>("method");
            auto prec = make_solver<F, NV, base_solver_interface>(method, A);
            prec->set_param_list(prms);
            prec->init();
            prec->setup(params);
            precond = prec;
        }
    }
    virtual void renew_params(const params::global_param_list &params, bool solver_mode = true) {
        if (!solver_mode)
            return;
        assert(params.find("solver"));
        param_list.override_params(params.get("solver"));
        if (!precond)
            return;
        if (params.find("preconditioner")) {
            precond->renew_param_list(params.get("preconditioner"));
            precond->renew_params(params, false);
        }
    }
    virtual void solve(vector::vector &_x, vector::vector &_b, const vector::vector &conv,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) {
        assert(initialized);
        x = &_x;
        b = &_b;
        solve(conv, tok);
    }

    virtual void solve(vector::vector &_x, vector::vector &_b,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) {
        assert(initialized);
        x = &_x;
        b = &_b;
        const vector::vector &conv = blas::ConstVectorsCache<F>::get_ones_vec(NV);
        solve(conv, tok);
    }

    virtual void solve(XAMG::mpi::token &tok = XAMG::mpi::null_token) {
        const vector::vector &conv = blas::ConstVectorsCache<F>::get_ones_vec(NV);
        solve(conv, tok);
    }

    virtual void solve(const vector::vector &conv, XAMG::mpi::token &tok = XAMG::mpi::null_token) {}
};

#define DECLARE_INHERITED_FROM_BASESOLVER(CLASSNAME)                                               \
    using base = base_solver<F, NV>;                                                               \
    using base_iface = base_solver_interface;                                                      \
    using base::stats;                                                                             \
    using base::param_list;                                                                        \
    using base::buffer;                                                                            \
    using base::A;                                                                                 \
    using base::x;                                                                                 \
    using base::b;                                                                                 \
    using base::precond;                                                                           \
    using base::allreduce_buffer;                                                                  \
    CLASSNAME(matrix::matrix &_A, vector::vector &_x, vector::vector &_b) : base(_A, _x, _b) {     \
        base::buffer_nvecs = nvecs;                                                                \
        base::allreduce_buffer_size = comm_size;                                                   \
    }                                                                                              \
    CLASSNAME(matrix::matrix &_A) : base(_A) {                                                     \
        base::buffer_nvecs = nvecs;                                                                \
        base::allreduce_buffer_size = comm_size;                                                   \
    }                                                                                              \
    CLASSNAME(const CLASSNAME &that) = delete;                                                     \
    CLASSNAME &operator=(const CLASSNAME &that) = delete;                                          \
    virtual ~CLASSNAME() {}                                                                        \
    virtual void solve(const vector::vector &conv, XAMG::mpi::token &tok = XAMG::mpi::null_token)  \
        override;                                                                                  \
    virtual void matrix_info() override;

template <typename F, uint16_t NV>
struct Identity : public base_solver<F, NV> {
    const uint16_t nvecs = 0;
    const uint16_t comm_size = 0;
    DECLARE_INHERITED_FROM_BASESOLVER(Identity)
};

template <typename F, uint16_t NV>
void Identity<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void Identity<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    assert(buffer.size() == (size_t)nvecs);
    vector::vector &x = *this->x;
    vector::vector &b = *this->b;
    uint16_t iters = 1;
    uint16_t progress = 0;
    bool flag = false;
    // if (tok != XAMG::mpi::null_token)
    //     XAMG::out << "Do progress!" << std::endl;
    for (int it = 0; it < iters; it++) {
        blas::copy<F, NV>(b, x);
        if ((tok != XAMG::mpi::null_token) && (!flag) && progress) {
            flag = XAMG::mpi::test(tok);
        }
    }
}

} // namespace solver
} // namespace XAMG

#include "detail/bicgstab.inl"
#include "detail/merged/merged_bicgstab.inl"
#include "detail/pbicgstab.inl"
#include "detail/merged/merged_pbicgstab.inl"
#include "detail/rbicgstab.inl"
#include "detail/merged/merged_rbicgstab.inl"
#include "detail/chebyshev.inl"
#include "detail/pcg.inl"
#include "detail/jacobi.inl"
#include "detail/hsgs.inl"
#include "detail/mg.inl"
#include "detail/direct.inl"

#ifdef XAMG_EXPERIMENTAL_SOLVERS
#include "detail/experimental/ibicgstab.inl"
#include "detail/experimental/merged/merged_ibicgstab.inl"
#include "detail/experimental/pipebicgstab.inl"
#include "detail/experimental/merged/merged_pipebicgstab.inl"
#include "detail/experimental/ppipebicgstab.inl"
#include "detail/experimental/merged/merged_ppipebicgstab.inl"
#endif

namespace XAMG {
namespace solver {

#define SOLVER_START_IF_CHAIN                                                                      \
    if (false)                                                                                     \
        ;
#define SOLVER_END_IF_CHAIN else assert(0 && "The selected method is not implemented");

#define SOLVER_NEW_OPERATOR(SOLVER)                                                                \
    else if (method == #SOLVER) if (with_vectors) return std::make_shared<SOLVER<F, NV>>(_A, *x,   \
                                                                                         *y);      \
    else return std::make_shared<SOLVER<F, NV>>(_A);

template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> internal_make_solver(const std::string &method, matrix::matrix &_A,
                                         XAMG::vector::vector *x, XAMG::vector::vector *y) {
    bool with_vectors = true;
    if (x == nullptr || y == nullptr)
        with_vectors = false;
    // std::string method;
    // params.get_value("method", method);

    SOLVER_START_IF_CHAIN
    SOLVER_NEW_OPERATOR(Identity)
    SOLVER_NEW_OPERATOR(MultiGrid)
    SOLVER_NEW_OPERATOR(Direct)
    SOLVER_NEW_OPERATOR(Jacobi)
    SOLVER_NEW_OPERATOR(HSGS)
    SOLVER_NEW_OPERATOR(Chebyshev)
    SOLVER_NEW_OPERATOR(PCG)
    SOLVER_NEW_OPERATOR(BiCGStab)
    SOLVER_NEW_OPERATOR(PBiCGStab)
    SOLVER_NEW_OPERATOR(RBiCGStab)
#ifdef XAMG_EXPERIMENTAL_SOLVERS
    SOLVER_NEW_OPERATOR(IBiCGStab)
    SOLVER_NEW_OPERATOR(PipeBiCGStab)
    SOLVER_NEW_OPERATOR(PPipeBiCGStab)
#endif
    SOLVER_NEW_OPERATOR(MergedBiCGStab)
    SOLVER_NEW_OPERATOR(MergedPBiCGStab)
    SOLVER_NEW_OPERATOR(MergedRBiCGStab)
#ifdef XAMG_EXPERIMENTAL_SOLVERS
    SOLVER_NEW_OPERATOR(MergedIBiCGStab)
    SOLVER_NEW_OPERATOR(MergedPipeBiCGStab)
    SOLVER_NEW_OPERATOR(MergedPPipeBiCGStab)
#endif
    SOLVER_END_IF_CHAIN
    return nullptr;
}

template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> make_solver(const std::string &method, matrix::matrix &_A,
                                XAMG::vector::vector &x, XAMG::vector::vector &y) {
    return internal_make_solver<F, NV, RT>(method, _A, &x, &y);
}

template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> make_solver(const std::string &method, matrix::matrix &_A) {
    return internal_make_solver<F, NV, RT>(method, _A, nullptr, nullptr);
}

template <typename F, uint16_t NV>
std::shared_ptr<base_solver_interface>
construct_basic_solver(const XAMG::params::global_param_list &params,
                       const XAMG::params::param_list &list, matrix::matrix &A) {
    auto method = list.get_value<std::string>("method");
    auto s = make_solver<F, NV, base_solver_interface>(method, A);
    s->set_param_list(list);
    s->init();
    s->setup(params);
    return s;
}

template <typename F, uint16_t NV>
std::shared_ptr<base_solver_interface>
construct_basic_solver(const XAMG::params::global_param_list &params,
                       const XAMG::params::param_list &list, matrix::matrix &A,
                       XAMG::vector::vector &x, XAMG::vector::vector &y) {
    auto method = list.get_value<std::string>("method");
    auto s = make_solver<F, NV, base_solver_interface>(method, A, x, y);
    s->set_param_list(list);
    s->init();
    s->setup(params);
    return s;
}

template <typename F, uint16_t NV>
std::shared_ptr<base_solver<F, NV>>
construct_solver_hierarchy(const params::global_param_list &params, matrix::matrix &A) {
    auto list = params.get("solver");
    auto method = list.get_value<std::string>("method");
    auto s = make_solver<F, NV, base_solver<F, NV>>(method, A);
    s->assemble(params);
    return s;
}

template <typename F, uint16_t NV>
std::shared_ptr<base_solver<F, NV>>
construct_solver_hierarchy(const params::global_param_list &params, matrix::matrix &A,
                           XAMG::vector::vector &x, XAMG::vector::vector &y) {
    auto list = params.get("solver");
    auto method = list.get_value<std::string>("method");
    auto s = make_solver<F, NV, base_solver<F, NV>>(method, A, x, y);
    s->assemble(params);
    return s;
}

#undef SOLVER_START_IF_CHAIN
#undef SOLVER_END_IF_CHAIN
#undef SOLVER_NEW_OPERATOR
#undef DECLARE_INHERITED_FROM_BASESOLVER
} // namespace solver
} // namespace XAMG
