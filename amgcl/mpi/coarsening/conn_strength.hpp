#ifndef AMGCL_MPI_COARSENING_CONN_STRENGTH_HPP
#define AMGCL_MPI_COARSENING_CONN_STRENGTH_HPP

/*
The MIT License

Copyright (c) 2012-2018 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/mpi/coarsening/conn_strength.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Strength of connection for the distributed_matrix.
 */

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {
namespace coarsening {

template <class Backend>
boost::shared_ptr< distributed_matrix< backend::builtin<bool> > >
conn_strength(const distributed_matrix<Backend> &A, float eps_strong) {
    AMGCL_TIC("strength");
    typedef typename Backend::value_type value_type;
    typedef typename math::scalar_of<value_type>::type scalar_type;
    typedef backend::crs<value_type> build_matrix;
    typedef backend::crs<bool> bool_matrix;

    communicator comm = A.comm();
    ptrdiff_t n = A.loc_rows();

    const build_matrix &A_loc = *A.local();
    const build_matrix &A_rem = *A.remote();
    const comm_pattern<Backend> &Ap = A.cpat();

    scalar_type eps_squared = eps_strong * eps_strong;

    boost::shared_ptr<bool_matrix> s_loc = boost::make_shared<bool_matrix>();
    boost::shared_ptr<bool_matrix> s_rem = boost::make_shared<bool_matrix>();

    bool_matrix &S_loc = *s_loc;
    bool_matrix &S_rem = *s_rem;

    S_loc.set_size(n, n, true);
    S_rem.set_size(n, 0, true);

    S_loc.val = new bool[A_loc.nnz];
    S_rem.val = new bool[A_rem.nnz];

    boost::shared_ptr< backend::numa_vector<value_type> > d = backend::diagonal(A_loc);
    backend::numa_vector<value_type> &D = *d;

    std::vector<value_type> D_loc(Ap.send.count());
    std::vector<value_type> D_rem(Ap.recv.count());

    for(size_t i = 0, nv = Ap.send.count(); i < nv; ++i)
        D_loc[i] = D[Ap.send.col[i]];

    Ap.exchange(&D_loc[0], &D_rem[0]);

#pragma omp parallel for
    for(ptrdiff_t i = 0; i < n; ++i) {
        value_type eps_dia_i = eps_squared * D[i];

        for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j) {
            ptrdiff_t  c = A_loc.col[j];
            value_type v = A_loc.val[j];

            if ((S_loc.val[j] = (c == i || (eps_dia_i * D[c] < v * v))))
                ++S_loc.ptr[i+1];
        }

        for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j) {
            ptrdiff_t  c = Ap.local_index(A_rem.col[j]);
            value_type v = A_rem.val[j];

            if ((S_rem.val[j] = (eps_dia_i * D_rem[c] < v * v)))
                ++S_rem.ptr[i+1];
        }
    }

    S_loc.col = new ptrdiff_t[S_loc.nnz = S_loc.scan_row_sizes()];
    S_rem.col = new ptrdiff_t[S_rem.nnz = S_rem.scan_row_sizes()];

#pragma omp parallel for
    for(ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t loc_head = S_loc.ptr[i];
        ptrdiff_t rem_head = S_rem.ptr[i];

        for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j)
            if (S_loc.val[j]) S_loc.col[loc_head++] = A_loc.col[j];

        for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j)
            if (S_rem.val[j]) S_rem.col[rem_head++] = A_rem.col[j];
    }
    AMGCL_TOC("strength");

    return boost::make_shared< distributed_matrix< backend::builtin<bool> > >(
            comm, s_loc, s_rem);
}

} // namespace coarsening
} // namespace mpi
} // namespace amgcl


#endif
