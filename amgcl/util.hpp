#ifndef AMGCL_UTIL_HPP
#define AMGCL_UTIL_HPP

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
 * \file   amgcl/util.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Various utilities.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <set>
#include <complex>
#include <limits>
#include <stdexcept>
#include <boost/property_tree/ptree.hpp>

/* Performance measurement macros
 *
 * If AMGCL_PROFILING macro is defined at compilation, then AMGCL_TIC(name) and
 * AMGCL_TOC(name) macros correspond to prof.tic(name) and prof.toc(name).
 * amgcl::prof should be an instance of amgcl::profiler<> defined in a user
 * code similar to:
 * \code
 * namespace amgcl { profiler<> prof; }
 * \endcode
 * If AMGCL_PROFILING is undefined, then AMGCL_TIC and AMGCL_TOC are noop macros.
 */
#ifdef AMGCL_PROFILING
#  include <amgcl/profiler.hpp>
#  define AMGCL_TIC(name) amgcl::prof.tic(name);
#  define AMGCL_TOC(name) amgcl::prof.toc(name);
namespace amgcl { extern profiler<> prof; }
#else
#  ifndef AMGCL_TIC
#    define AMGCL_TIC(name)
#  endif
#  ifndef AMGCL_TOC
#    define AMGCL_TOC(name)
#  endif
#endif

#define AMGCL_DEBUG_SHOW(x)                                                    \
    std::cout << std::setw(20) << #x << ": "                                   \
              << std::setw(15) << std::setprecision(8) << std::scientific      \
              << (x) << std::endl

namespace amgcl {

/// Throws \p message if \p condition is not true.
template <class Condition, class Message>
void precondition(const Condition &condition, const Message &message) {
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4800)
#endif
    if (!condition) throw std::runtime_error(message);
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
}

#define AMGCL_PARAMS_IMPORT_VALUE(p, name)                                     \
    name( p.get(#name, params().name) )

#define AMGCL_PARAMS_IMPORT_CHILD(p, name)                                     \
    name( p.get_child(#name, amgcl::detail::empty_ptree()) )

#define AMGCL_PARAMS_EXPORT_VALUE(p, path, name)                               \
    p.put(std::string(path) + #name, name)

#define AMGCL_PARAMS_EXPORT_CHILD(p, path, name)                               \
    name.get(p, std::string(path) + #name + ".")

// Missing parameter action
#ifndef AMGCL_PARAM_MISSING
#  define AMGCL_PARAM_MISSING(name) (void)0
#endif

// Unknown parameter action
#ifndef AMGCL_PARAM_UNKNOWN
#  define AMGCL_PARAM_UNKNOWN(name)                                            \
      std::cerr << "AMGCL WARNING: unknown parameter " << name << std::endl
#endif

inline void check_params(
        const boost::property_tree::ptree &p,
        const std::set<std::string> &names
        )
{
    for(const auto &n : names) {
        if (!p.count(n)) {
            AMGCL_PARAM_MISSING(n);
        }
    }
    for(const auto &v : p) {
        if (!names.count(v.first)) {
            AMGCL_PARAM_UNKNOWN(v.first);
        }
    }
}

inline void check_params(
        const boost::property_tree::ptree &p,
        const std::set<std::string> &names,
        const std::set<std::string> &opt_names
        )
{
    for(const auto &n : names) {
        if (!p.count(n)) {
            AMGCL_PARAM_MISSING(n);
        }
    }
    for(const auto &n : opt_names) {
        if (!p.count(n)) {
            AMGCL_PARAM_MISSING(n);
        }
    }
    for(const auto &v : p) {
        if (!names.count(v.first) && !opt_names.count(v.first)) {
            AMGCL_PARAM_UNKNOWN(v.first);
        }
    }
}

// Put parameter in form "key=value" into a boost::property_tree::ptree
inline void put(boost::property_tree::ptree &p, const std::string &param) {
    size_t eq_pos = param.find('=');
    if (eq_pos == std::string::npos)
        throw std::invalid_argument("param in amgcl::put() should have \"key=value\" format!");
    p.put(param.substr(0, eq_pos), param.substr(eq_pos + 1));
}

namespace detail {

inline const boost::property_tree::ptree& empty_ptree() {
    static const boost::property_tree::ptree p;
    return p;
}

struct empty_params {
    empty_params() {}
    empty_params(const boost::property_tree::ptree &p) {
        for(boost::property_tree::ptree::const_iterator v = p.begin(), e = p.end(); v != e; ++v)
            AMGCL_PARAM_UNKNOWN(v->first);
    }
    void get(boost::property_tree::ptree&, const std::string&) const {}
};

template <class T>
T eps(size_t n) {
    return 2 * std::numeric_limits<T>::epsilon() * n;
}

} // namespace detail

template <class T> struct is_complex : std::false_type {};
template <class T> struct is_complex< std::complex<T> > : std::true_type {};
} // namespace amgcl

namespace std {

// Read pointers from input streams.
// This allows to exchange pointers through boost::property_tree::ptree.
template <class T>
inline istream& operator>>(istream &is, T* &ptr) {
    std::ios_base::fmtflags ff(is.flags());

    size_t val;
    is >> std::hex >> val;

    ptr = reinterpret_cast<T*>(val);

    is.flags(ff);
    return is;
}

} // namespace std


#endif
