#pragma once

#include <vector>
#include <array>
#include <cassert>

#include "mesh.hpp"
#include "mesh_storage.hpp"

#include "ident.hpp"
#include "point.hpp"

namespace hho {

namespace priv {

template<size_t DIM, size_t CODIM>
struct howmany;

}

template<size_t DIM, size_t CODIM>
class simplicial_element
{
    typedef point_identifier<DIM>       point_id_type;

    typedef std::array<point_id_type, priv::howmany<DIM, CODIM>::nodes> node_array_type;

    node_array_type     m_pts_ptrs;

public:
    typedef identifier<simplicial_element, ident_impl_t, 0> id_type;

    simplicial_element() = default;

    simplicial_element(std::initializer_list<point_id_type> l)
    {
        std::copy(l.begin(), l.end(), m_pts_ptrs.begin());
        std::sort(m_pts_ptrs.begin(), m_pts_ptrs.end());
    }

    node_array_type point_ids(void) const
    {
        return m_pts_ptrs;
    }

    bool operator<(const simplicial_element& other) const
    {
        return m_pts_ptrs < other.m_pts_ptrs;
    }

    bool operator==(const simplicial_element& other) const
    {
        return m_pts_ptrs == other.m_pts_ptrs;
    }
};

namespace priv {
    template<size_t DIM>
    struct howmany<DIM, DIM>
    {
        static const size_t nodes = 1;
        static const size_t edges = 0;
        static const size_t surfaces = 0;
        static const size_t volumes = 0;
    };

    template<>
    struct howmany<3,0>
    {
        static const size_t nodes = 4;
        static const size_t edges = 6;
        static const size_t surfaces = 4;
        static const size_t volumes = 1;
        static const size_t subelements = surfaces;
    };

    template<>
    struct howmany<3,1>
    {
        static const size_t nodes = 3;
        static const size_t edges = 3;
        static const size_t surfaces = 1;
        static const size_t volumes = 0;
        static const size_t subelements = edges;
    };

    template<>
    struct howmany<3,2>
    {
        static const size_t nodes = 2;
        static const size_t edges = 1;
        static const size_t surfaces = 0;
        static const size_t volumes = 0;
        static const size_t subelements = nodes;
    };

    template<>
    struct howmany<2,0>
    {
        static const size_t nodes = 3;
        static const size_t edges = 3;
        static const size_t surfaces = 1;
        static const size_t volumes = 0;
        static const size_t subelements = edges;
    };

    template<>
    struct howmany<2,1>
    {
        static const size_t nodes = 2;
        static const size_t edges = 1;
        static const size_t surfaces = 0;
        static const size_t volumes = 0;
        static const size_t subelements = nodes;
    };

    template<>
    struct howmany<1,0>
    {
        static const size_t nodes = 2;
        static const size_t edges = 1;
        static const size_t surfaces = 0;
        static const size_t volumes = 0;
        static const size_t subelements = nodes;
    };
} // namespace priv

} // namespace hho

/* Output streaming operator for elements */
template<size_t DIM, size_t CODIM>
std::ostream&
operator<<(std::ostream& os, const hho::simplicial_element<DIM, CODIM>& e)
{
    os << "simplicial_element<" << DIM << "," << CODIM << ">: ";
    auto pts = e.point_ids();
    for (auto itor = pts.begin(); itor != pts.end(); itor++)
        os << *itor << " ";

    return os;
}
