/* HHO code - N-dimensional point
 *
 * HISTORY:
 *  - 08/02/2016 (mc): File created
 *
 */

#pragma once

#include <iostream>
#include <array>
#include <stdexcept>
#include <initializer_list>

#include "eigen.hpp"
#include "ident.hpp"

template<typename T, size_t DIM>
class point
{
    static_vector<T, DIM>     m_coords;

public:
    typedef identifier<point, ident_impl_t, 0>  id_type;

    point()
    {
        m_coords = static_vector<T, DIM>::Zero(DIM);
    }

    point(const point& other) : m_coords(other.m_coords) {}

    point(std::initializer_list<T> l)
    {
        if (l.size() != DIM)
            throw std::invalid_argument("Wrong initializer list size");

        for (size_t i = 0; i < DIM; i++)
            m_coords(i) = *(l.begin()+i);

    }

    T   at(size_t pos) const
    {
        if (pos >= DIM)
            throw std::out_of_range("access out of range");

        return m_coords(pos);
    }

    T&  at(size_t pos)
    {
        if (pos >= DIM)
            throw std::out_of_range("access out of range");

        return m_coords(pos);
    }

    T   operator[](size_t pos) const { return m_coords(pos); }
    T&  operator[](size_t pos)       { return m_coords(pos); }

    template<typename U = T>
    typename std::enable_if<DIM == 1 || DIM == 2 || DIM == 3, U>::type
    x() const { return m_coords(0); }

    template<typename U = T>
    typename std::enable_if<DIM == 1 || DIM == 2 || DIM == 3, U>::type&
    x() { return m_coords(0); }

    template<typename U = T>
    typename std::enable_if<DIM == 2 || DIM == 3, U>::type
    y() const { return m_coords(1); }

    template<typename U = T>
    typename std::enable_if<DIM == 2 || DIM == 3, U>::type&
    y() { return m_coords(1); }

    template<typename U = T>
    typename std::enable_if<DIM == 3, U>::type
    z() const { return m_coords(2); }

    template<typename U = T>
    typename std::enable_if<DIM == 3, U>::type&
    z() { return m_coords(2); }

    auto to_vector() const
    {
        return m_coords;
    }

    friend point operator+(const point& p1, const point& p2)
    {
        point ret;
        ret.m_coords = p1.m_coords + p2.m_coords;
        return ret;
    }

    friend point operator-(const point& p1, const point& p2)
    {
        point ret;
        ret.m_coords = p1.m_coords - p2.m_coords;
        return ret;
    }

    friend point operator*(const point& p, T scalefactor)
    {
        point ret;
        ret.m_coords = p.m_coords * scalefactor;
        return ret;
    }

    friend point operator*(T scalefactor, const point& p)
    {
        return p * scalefactor;
    }

    friend point operator/(const point& p, T scalefactor)
    {
        point ret;
        ret.m_coords = p.m_coords / scalefactor;
        return ret;
    }
};

template<typename T, size_t DIM>
std::ostream&
operator<<(std::ostream& os, const point<T, DIM>& pt)
{
    os << "( ";
    for (size_t i = 0; i < DIM; i++)
    {
        os << pt[i];

        if (i < DIM-1)
            os << ", ";
    }
    os << " )";
    return os;
}

struct dummy_point_type {};
template<size_t DIM>
using point_identifier = identifier<point<dummy_point_type, DIM>, ident_impl_t, 0>;
