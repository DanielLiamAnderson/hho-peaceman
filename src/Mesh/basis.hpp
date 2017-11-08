#pragma once

#include <vector>
#include "eigen.hpp"
#include "util.h"

template<typename T, typename ElemType>
class scalar_polynomial_basis;

template<typename T, template<size_t, size_t> class ElemType>
class scalar_polynomial_basis<T, ElemType<2, 1>>
{
    typedef ElemType<2,1> elem_type;

    T               m_hF;
    point<T,2>      m_p0, m_pF;
    size_t          m_degree;

    T phi(size_t k, const point<T,2>& x) const
    {
        auto u = (x-m_pF).to_vector();
        auto v = (m_p0-m_pF).to_vector();
        T d = u.dot(v)/(m_hF*m_hF);
        return pow(d,k);
    }

public:
    scalar_polynomial_basis()
        : m_hF(0), m_degree(0)
    {}

    template<typename MeshType>
    scalar_polynomial_basis(const MeshType& msh, const elem_type& elem, size_t degree)
    {
        static_assert(std::is_same<typename MeshType::face, elem_type>::value,
                        "Wrong mesh or wrong instantiation of scalar_polynomial_basis");

        m_hF = measure(msh, elem);
        auto pts = points(msh, elem);
        assert(pts.size() == 2);
        m_p0 = pts[0];
        m_pF = barycenter(msh, elem);
        m_degree    = degree;
    }

    static size_t degree_index(size_t degree)
    {
        return degree;
    }

    static size_t basis_size(size_t degree)
    {
        return degree+1;
    }

    std::vector<T>
    eval_from_to(size_t from, size_t to, const point<T,2>& p) const
    {
        std::vector<T> bv;
        bv.reserve( m_degree );

        for (size_t k = from; k <= to; k++)
                bv.push_back( phi(k, p) );

        return bv;
    }

    std::vector<T>
    eval(const point<T,2>& p) const
    {
        return eval_from_to(0, m_degree, p);
    }
};

template<typename T, template<size_t, size_t> class ElemType>
class scalar_polynomial_basis<T, ElemType<2, 0>>
{
    typedef ElemType<2,0> elem_type;

    T                       m_hT;
    point<T,2>              m_pT;
    size_t                  m_degree;

    T phi(size_t i, size_t k, const point<T,2>& p) const
    {
        auto np = (p - m_pT)/m_hT;
        return pow(np.x(), i) * pow(np.y(), k-i);
    }

    static_vector<T,2> dphi(size_t i, size_t k, const point<T,2>& p) const
    {
        static_vector<T,2> g;
        auto np = (p - m_pT)/m_hT;
        g(0) = (i == 0) ? 0 : i * pow(np.x(),i-1) / m_hT * pow(np.y(),k-i);
        g(1) = (i == k) ? 0 : pow(np.x(),i) * (k-i) * pow(np.y(),k-i-1) / m_hT;
        return g;
    }

public:
    scalar_polynomial_basis()
        : m_hT(0), m_degree(0)
    {}

    template<typename MeshType>
    scalar_polynomial_basis(const MeshType& msh, const elem_type& elem, size_t degree)
    {
        static_assert(std::is_same<typename MeshType::cell, elem_type>::value,
                        "Wrong mesh or wrong instantiation of scalar_polynomial_basis");
        m_hT        = diameter(msh, elem);
        m_pT        = barycenter(msh, elem);
        m_degree    = degree;
    }

    static size_t degree_index(size_t k)
    {
        return binomial(k+1, 2);
    }

    static size_t basis_size(size_t degree)
    {
        return binomial(degree+2, 2);
    }

    std::vector<T>
    eval_from_to(size_t from, size_t to, const point<T,2>& p) const
    {
        std::vector<T> bv;
        bv.reserve( binomial(m_degree+2, 2) );

        // 1, y, x, y^2, xy, x^2, ...
        for (size_t k = from; k <= to; k++)
            for (size_t i = 0; i <= k; i++)
                bv.push_back( phi(i, k, p) );

        return bv;
    }

    std::vector<T>
    eval(const point<T,2>& p) const
    {
        return eval_from_to(0, m_degree, p);
    }

    std::vector<static_vector<T,2>>
    eval_gradient_from_to(size_t from, size_t to, const point<T,2>& p) const
    {
        std::vector<static_vector<T,2>> gv;
        gv.reserve( binomial(m_degree+2, 2) );

        // 1, y, x, y^2, xy, x^2, ...
        for (size_t k = from; k <= to; k++)
            for (size_t i = 0; i <= k; i++)
                gv.push_back( dphi(i, k, p) );

        return gv;
    }

    std::vector<static_vector<T,2>>
    eval_gradient(const point<T,2>& p) const
    {
        return eval_gradient_from_to(0, m_degree, p);
    }

};
