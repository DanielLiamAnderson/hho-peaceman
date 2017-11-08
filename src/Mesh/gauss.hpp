#pragma once

#include <vector>
#include "Mesh/point.hpp"
#include "Mesh/eigen.hpp"
#include "Mesh/contrib/triangle_dunavant_rule.hpp"

unsigned int fact(unsigned int n)
{
    return (n < 2) ? 1 : n*fact(n-1);
}

unsigned int binomial(unsigned int n, unsigned int k)
{
    return fact(n)/(fact(n-k)*fact(k));
}

/* See for example 'Gaussian Quadrature and the Eigenvalue Problem' by
 * John A. Gubner.
 */
template<typename T>
std::pair<dynamic_vector<T>, dynamic_vector<T>>
gauss_quadrature(size_t doe)
{
    using namespace Eigen;

    if (doe%2 == 0)
        doe++;

    size_t num_nodes = (doe+1)/2;

    if (num_nodes == 1)
    {
        dynamic_vector<T> weights;
        weights.resize(1);
        weights(0) = 1.0;

        dynamic_vector<T> nodes;
        nodes.resize(1);
        nodes(0) = 0.5;

        return std::make_pair(nodes, weights);
    }

    dynamic_matrix<T> M = dynamic_matrix<T>::Zero(num_nodes, num_nodes);
    for (size_t i = 1; i < num_nodes; i++)
    {
        T p = 4.0 - 1.0/(i*i);
        M(i, i-1) = sqrt(1.0/p);
    }

    SelfAdjointEigenSolver<dynamic_matrix<T>> solver;
    solver.compute(M);

    dynamic_vector<T> weights = solver.eigenvectors().row(0);
                      weights = weights.array().square();
    dynamic_vector<T> nodes = solver.eigenvalues();

    for (int i = 0; i < nodes.size(); i++)
        nodes(i) = (nodes(i) + 1.)/2.;

    assert( weights.size() == nodes.size() );

    return std::make_pair(nodes, weights);
}

template<typename T>
std::pair<std::vector<point<T,2>>, std::vector<T>>
dunavant_quadrature(int doe)
{
    static_assert(std::is_same<T, double>::value,
                  "code computing dunavant triangulations is crap and supports only double");
    std::vector<point<T,2>>     nodes;
    std::vector<T>              weights;

    int rule_num = dunavant_rule_num();
    int rule, order_num, degree;

    for(rule = 1; rule <= rule_num; rule++)
    {
        degree = dunavant_degree(rule);
        if(degree >= doe)
            break;
    }
    assert(rule != rule_num or degree >= doe);

    order_num = dunavant_order_num(rule);
    std::vector<T> xytab(2*order_num), wtab(order_num);

    dunavant_rule(rule, order_num, &xytab[0], &wtab[0]);

    nodes.resize(order_num);
    weights.resize(order_num);
    for(int iQN = 0; iQN < order_num; iQN++)
    {
        point<T,2> xQN;
        xQN.at(0) = xytab[0+iQN*2];
        xQN.at(1) = xytab[1+iQN*2];
        nodes.at(iQN) = xQN;
        weights.at(iQN) = wtab[iQN];
    }

    return std::make_pair(nodes, weights);
}


template<typename T, size_t DIM>
class hierarchical_scalar_basis;

template<typename T>
class hierarchical_scalar_basis<T, 1>
{
    point<T,2>  m_x0, m_xF;
    T           m_hF;

public:
    hierarchical_scalar_basis() = default;

    hierarchical_scalar_basis(const point<T,2> x0, const point<T,2> xF, T hF)
        : m_x0(x0), m_xF(xF), m_hF(hF)
    {}

    T phi(size_t k, const point<T,2>& x)
    {
        auto u = (x-m_xF).to_vector();
        auto v = (m_x0-m_xF).to_vector();
        T d = u.dot(v)/m_hF;
        return pow(d,k);
    }
};

template<typename T>
class hierarchical_scalar_basis<T, 2>
{
    T           m_hT;
    point<T,2>  m_pT;
    size_t      m_degree;

public:
    T phi(size_t i, size_t k, const point<T,2>& p)
    {
        auto np = (p - m_pT)/m_hT;
        return pow(np.x(), i) * pow(np.y(), k-i);
    }

    static_vector<T,2> dphi(size_t i, size_t k, const point<T,2>& p)
    {
        static_vector<T,2> g;
        auto np = (p - m_pT)/m_hT;
        g(0) = (i == 0) ? 0 : i * pow(np.x(),i-1) *         pow(np.y(),k-i)   / m_hT;
        g(1) = (i == k) ? 0 :    pow(np.x(),i)   * (k-i) * pow(np.y(),k-i-1) / m_hT;
        return g;
    }

    size_t size() const
    {
        return binomial(m_degree + 1 + 2, 2);
    }

    hierarchical_scalar_basis() = default;

    hierarchical_scalar_basis(T hT, point<T,2> p, size_t degree)
        : m_hT(hT), m_pT(p), m_degree(degree)
    {}

};

