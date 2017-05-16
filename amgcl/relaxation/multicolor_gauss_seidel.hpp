#ifndef AMGCL_RELAXATION_MULTICOLOR_GAUSS_SEIDEL_HPP
#define AMGCL_RELAXATION_MULTICOLOR_GAUSS_SEIDEL_HPP

/*
The MIT License

Copyright (c) 2012-2017 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   amgcl/relaxation/multicolor_gauss_seidel.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Multicolor Gauss-Seidel relaxation scheme.
 */

#include <algorithm>
#include <numeric>

#include <amgcl/backend/interface.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace relaxation {

template <class Backend>
struct multicolor_gauss_seidel {
    typedef amgcl::detail::empty_params params;

    template <class Matrix>
    multicolor_gauss_seidel(
            const Matrix &A,
            const params&,
            const typename Backend::params&
            )
        : forward(A), backward(A)
    {
    }

    template <class Matrix, class VecRHS, class VecX, class VecTMP>
    void apply_pre(const Matrix&, const VecRHS &rhs, VecX &x, VecTMP&, const params&) const
    {
        forward.sweep(rhs, x);
    }

    template <class Matrix, class VecRHS, class VecX, class VecTMP>
    void apply_post(const Matrix&, const VecRHS &rhs, VecX &x, VecTMP&, const params&) const
    {
        backward.sweep(rhs, x);
    }

    template <class Matrix, class VecRHS, class VecX>
    void apply(const Matrix&, const VecRHS &rhs, VecX &x, const params&) const
    {
        backend::clear(x);
        forward.sweep(rhs, x);
        backward.sweep(rhs, x);
    }

    private:
        static int num_threads() {
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        }

        static int thread_id() {
#ifdef _OPENMP
            return omp_get_thread_num();
#else
            return 0;
#endif
        }

        template <bool forward>
        struct gs_sweep {
            typedef typename Backend::value_type value_type;
            typedef typename Backend::rhs_type   rhs_type;

            struct task {
                ptrdiff_t beg, end;
                task(ptrdiff_t beg, ptrdiff_t end) : beg(beg), end(end) {}
            };

            int nthreads;

            // thread-specific storage:
            std::vector< std::vector<task>       > tasks;
            std::vector< std::vector<ptrdiff_t>  > ptr;
            std::vector< std::vector<ptrdiff_t>  > col;
            std::vector< std::vector<value_type> > val;
            std::vector< std::vector<ptrdiff_t>  > ord;

            template <class Matrix>
            gs_sweep(const Matrix &A)
                : nthreads(num_threads()), tasks(nthreads),
                  ptr(nthreads), col(nthreads), val(nthreads), ord(nthreads)
            {
                typedef typename backend::row_iterator<Matrix>::type row_iterator;

                ptrdiff_t n    = backend::rows(A);
                ptrdiff_t nlev = 0;

                std::vector<ptrdiff_t> level(n, 0);
                std::vector<ptrdiff_t> order(n, 0);

                // 1. split rows into levels.
                ptrdiff_t beg = forward ? 0 : n-1;
                ptrdiff_t end = forward ? n :  -1;
                ptrdiff_t inc = forward ? 1 :  -1;

                for(ptrdiff_t i = beg; i != end; i += inc) {
                    ptrdiff_t l = level[i];

                    for(row_iterator a = row_begin(A, i); a; ++a) {
                        ptrdiff_t c = a.col();

                        if (forward) {
                            if (c >= i) continue;
                        } else {
                            if (c <= i) continue;
                        }

                        l = std::max(l, level[c]+1);
                    }

                    level[i] = l;
                    nlev = std::max(nlev, l+1);
                }


                // 2. reorder matrix rows.
                std::vector<ptrdiff_t> start(nlev+1, 0);

                for(ptrdiff_t i = 0; i < n; ++i)
                    ++start[level[i]+1];

                std::partial_sum(start.begin(), start.end(), start.begin());

                for(ptrdiff_t i = 0; i < n; ++i)
                    order[start[level[i]]++] = i;

                std::rotate(start.begin(), start.end() - 1, start.end());
                start[0] = 0;


                // 3. Organize matrix rows into tasks.
                //    Each level is split into nthreads tasks.
                std::vector<ptrdiff_t> thread_rows(nthreads, 0);
                std::vector<ptrdiff_t> thread_cols(nthreads, 0);

#pragma omp parallel
                {
                    int tid = thread_id();
                    tasks[tid].reserve(nlev);

                    for(ptrdiff_t lev = 0; lev < nlev; ++lev) {
                        // split each level into tasks.
                        ptrdiff_t lev_size = start[lev+1] - start[lev];
                        ptrdiff_t chunk_size = (lev_size + nthreads - 1) / nthreads;

                        ptrdiff_t beg = std::min(tid * chunk_size, lev_size);
                        ptrdiff_t end = std::min(beg + chunk_size, lev_size);

                        beg += start[lev];
                        end += start[lev];

                        tasks[tid].push_back(task(beg, end));

                        // count rows and nonzeros in the current task
                        thread_rows[tid] += end - beg;
                        for(ptrdiff_t i = beg; i < end; ++i) {
                            ptrdiff_t j = order[i];
                            thread_cols[tid] += row_nonzeros(A, j);
                        }
                    }
                }

                // 4. reorganize matrix data for better cache and NUMA locality.
#pragma omp parallel
                {
                    int tid = thread_id();

                    col[tid].reserve(thread_cols[tid]);
                    val[tid].reserve(thread_cols[tid]);
                    ord[tid].reserve(thread_rows[tid]);
                    ptr[tid].reserve(thread_rows[tid] + 1);
                    ptr[tid].push_back(0);

                    BOOST_FOREACH(task &t, tasks[tid]) {
                        ptrdiff_t loc_beg = ptr[tid].size() - 1;
                        ptrdiff_t loc_end = loc_beg;

                        for(ptrdiff_t r = t.beg; r < t.end; ++r, ++loc_end) {
                            ptrdiff_t i = order[r];

                            ord[tid].push_back(i);

                            for(row_iterator a = row_begin(A, i); a; ++a) {
                                col[tid].push_back(a.col());
                                val[tid].push_back(a.value());
                            }

                            ptr[tid].push_back(col[tid].size());
                        }

                        t.beg = loc_beg;
                        t.end = loc_end;
                    }
                }
            }

            template <class Vector1, class Vector2>
            void sweep(const Vector1 &rhs, Vector2 &x) const {
#pragma omp parallel
                {
                    int tid = thread_id();

                    BOOST_FOREACH(const task &t, tasks[tid]) {
                        for(ptrdiff_t r = t.beg; r < t.end; ++r) {
                            ptrdiff_t i   = ord[tid][r];
                            ptrdiff_t beg = ptr[tid][r];
                            ptrdiff_t end = ptr[tid][r+1];

                            rhs_type   X = rhs[i];
                            value_type D = math::identity<value_type>();

                            for(ptrdiff_t j = beg; j < end; ++j) {
                                ptrdiff_t  c = col[tid][j];
                                value_type v = val[tid][j];

                                if (c == i)
                                    D = v;
                                else
                                    X -= v * x[c];
                            }

                            x[i] = math::inverse(D) * X;
                        }

                        // each task corresponds to a level, so we need
                        // to synchronize across threads at this point:
#pragma omp barrier
                    }
                }
            }
        };

        gs_sweep<true > forward;
        gs_sweep<false> backward;
};

} // namespace relaxation

namespace backend {

template <class Backend>
struct relaxation_is_supported<
    Backend,
    relaxation::multicolor_gauss_seidel,
    typename boost::disable_if<
            typename Backend::provides_row_iterator
        >::type
    > : boost::false_type
{};

} // namespace backend
} // namespace amgcl


#endif
