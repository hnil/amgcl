#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/range/iterator_range.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/adapter/zero_copy.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl { profiler<> prof; }
using amgcl::prof;

typedef amgcl::scoped_tic< amgcl::profiler<> > scoped_tic;

//---------------------------------------------------------------------------
template <int B, template <class> class Precond>
boost::tuple<size_t, double> block_solve(
        const boost::property_tree::ptree &prm,
        size_t rows,
        std::vector<ptrdiff_t> const &ptr,
        std::vector<ptrdiff_t> const &col,
        std::vector<double>    const &val,
        std::vector<double>    const &rhs,
        std::vector<double>          &x
        )
{
    typedef amgcl::static_matrix<double, B, B> value_type;
    typedef amgcl::static_matrix<double, B, 1> rhs_type;
    typedef amgcl::backend::builtin<value_type> Backend;

    typedef amgcl::make_solver<
        Precond<Backend>,
        amgcl::runtime::iterative_solver<Backend>
        > Solver;

    prof.tic("setup");
    Solver solve(amgcl::adapter::block_matrix<B, value_type>(boost::tie(rows, ptr, col, val)), prm);
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    {
        scoped_tic t(prof, "solve");

        rhs_type const * fptr = reinterpret_cast<rhs_type const *>(&rhs[0]);
        rhs_type       * xptr = reinterpret_cast<rhs_type       *>(&x[0]);

        boost::iterator_range<rhs_type const *> frng(fptr, fptr + rows/B);
        boost::iterator_range<rhs_type       *> xrng(xptr, xptr + rows/B);

        return solve(frng, xrng);
    }
}

//---------------------------------------------------------------------------
template <template <class> class Precond>
boost::tuple<size_t, double> scalar_solve(
        const boost::property_tree::ptree &prm,
        size_t rows,
        std::vector<ptrdiff_t> const &ptr,
        std::vector<ptrdiff_t> const &col,
        std::vector<double>    const &val,
        std::vector<double>    const &rhs,
        std::vector<double>          &x
        )
{
    typedef amgcl::backend::builtin<double> Backend;

    typedef amgcl::make_solver<
        Precond<Backend>,
        amgcl::runtime::iterative_solver<Backend>
        > Solver;

    prof.tic("setup");
    Solver solve(amgcl::adapter::zero_copy(rows, &ptr[0], &col[0], &val[0]), prm);
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    {
        scoped_tic t(prof, "solve");

        return solve(rhs, x);
    }
}

int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    using amgcl::prof;
    using amgcl::precondition;
    using std::vector;
    using std::string;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help.")
        ("prm-file,P",
         po::value<string>(),
         "Parameter file in json format. "
        )
        (
         "prm,p",
         po::value< vector<string> >(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
        )
        ("matrix,A",
         po::value<string>(),
         "System matrix in the MatrixMarket format. "
         "When not specified, solves a Poisson problem in 3D unit cube. "
        )
        (
         "rhs,f",
         po::value<string>(),
         "The RHS vector in the MatrixMarket format. "
         "When omitted, a vector of ones is used by default. "
         "Should only be provided together with a system matrix. "
        )
        (
         "null,N",
         po::value<string>(),
         "The near null-space vectors in the MatrixMarket format. "
         "Should be a dense matrix of size N*M, where N is the number of "
         "unknowns, and M is the number of null-space vectors. "
         "Should only be provided together with a system matrix. "
        )
        (
         "block-size,b",
         po::value<int>()->default_value(1),
         "The block size of the system matrix. "
         "When specified, the system matrix is assumed to have block-wise structure. "
         "This usually is the case for problems in elasticity, structural mechanics, "
         "for coupled systems of PDE (such as Navier-Stokes equations), etc. "
         "Valid choices are 2, 3, 4, and 6."
        )
        (
         "size,n",
         po::value<int>()->default_value(32),
         "The size of the Poisson problem to solve when no system matrix is given. "
         "Specified as number of grid nodes along each dimension of a unit cube. "
         "The resulting system will have n*n*n unknowns. "
        )
        (
         "single-level,1",
         po::bool_switch()->default_value(false),
         "When specified, the AMG hierarchy is not constructed. "
         "Instead, the problem is solved using a single-level smoother as preconditioner. "
        )
        (
         "initial,x",
         po::value<double>()->default_value(0),
         "Value to use as initial approximation. "
        )
        (
         "output,o",
         po::value<string>(),
         "Output file. Will be saved in the MatrixMarket format. "
         "When omitted, the solution is not saved. "
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    boost::property_tree::ptree prm;
    if (vm.count("prm-file")) {
        read_json(vm["prm-file"].as<string>(), prm);
    }

    if (vm.count("prm")) {
        BOOST_FOREACH(string pair, vm["prm"].as<vector<string> >())
        {
            using namespace boost::algorithm;
            vector<string> key_val;
            split(key_val, pair, is_any_of("="));
            if (key_val.size() != 2) throw po::invalid_option_value(
                    "Parameters specified with -p option "
                    "should have name=value format");

            prm.put(key_val[0], key_val[1]);
        }
    }

    size_t rows;
    vector<ptrdiff_t> ptr, col;
    vector<double> val, rhs, null, x;

    if (vm.count("matrix")) {
        scoped_tic t(prof, "reading");

        using namespace amgcl::io;

        size_t cols;
        boost::tie(rows, cols) = mm_reader(vm["matrix"].as<string>())(
                ptr, col, val);

        precondition(rows == cols, "Non-square system matrix");

        if (vm.count("rhs")) {
            precondition(
                    boost::make_tuple(rows, 1) == mm_reader(vm["rhs"].as<string>())(rhs),
                    "The RHS vector has wrong size"
                    );
        } else {
            rhs.resize(rows, 1.0);
        }

        if (vm.count("null")) {
            size_t m, nv;
            boost::tie(m, nv) = mm_reader(vm["null"].as<string>())(null);
            precondition(m == rows, "Near null-space vectors have wrong size");

            prm.put("precond.coarsening.nullspace.cols", nv);
            prm.put("precond.coarsening.nullspace.rows", rows);
            prm.put("precond.coarsening.nullspace.B",    &null[0]);
        }
    } else {
        scoped_tic t(prof, "assembling");

        rows = sample_problem(vm["size"].as<int>(), val, col, ptr, rhs);
    }

    x.resize(rows, vm["initial"].as<double>());

    size_t iters;
    double error;

    int block_size    = vm["block-size"].as<int>();
    bool single_level = vm["single-level"].as<bool>();

    if (single_level) {
        switch (block_size) {
            case 1:
                boost::tie(iters, error) = scalar_solve<amgcl::runtime::relaxation::as_preconditioner>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            case 2:
                boost::tie(iters, error) = block_solve<2, amgcl::runtime::relaxation::as_preconditioner>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            case 3:
                boost::tie(iters, error) = block_solve<3, amgcl::runtime::relaxation::as_preconditioner>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            case 4:
                boost::tie(iters, error) = block_solve<4, amgcl::runtime::relaxation::as_preconditioner>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            case 6:
                boost::tie(iters, error) = block_solve<6, amgcl::runtime::relaxation::as_preconditioner>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            default:
                precondition(false, "Unsupported block size");
        }
    } else {
        switch (block_size) {
            case 1:
                boost::tie(iters, error) = scalar_solve<amgcl::runtime::amg>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            case 2:
                boost::tie(iters, error) = block_solve<2, amgcl::runtime::amg>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            case 3:
                boost::tie(iters, error) = block_solve<3, amgcl::runtime::amg>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            case 4:
                boost::tie(iters, error) = block_solve<4, amgcl::runtime::amg>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            case 6:
                boost::tie(iters, error) = block_solve<6, amgcl::runtime::amg>(
                        prm, rows, ptr, col, val, rhs, x);
                break;
            default:
                precondition(false, "Unsupported block size");
        }
    }

    if (vm.count("output")) {
        scoped_tic t(prof, "write");
        amgcl::io::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl
              << prof << std::endl;
}
