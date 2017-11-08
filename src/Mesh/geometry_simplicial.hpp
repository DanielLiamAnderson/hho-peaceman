#pragma once

#include "mesh.hpp"
#include "element_simplicial.hpp"

namespace hho {

namespace simplicial_priv {
template<size_t DIM>
struct element_types
{
    static_assert(DIM > 0 && DIM <= 3, "element_types: CODIM must be less than DIM");
};

template<>
struct element_types<3> {
        typedef simplicial_element<3,0>    volume_type;
        typedef simplicial_element<3,1>    surface_type;
        typedef simplicial_element<3,2>    edge_type;
        typedef simplicial_element<3,3>    node_type;
};

template<>
struct element_types<2> {
        typedef simplicial_element<2,0>    surface_type;
        typedef simplicial_element<2,1>    edge_type;
        typedef simplicial_element<2,2>    node_type;
};

template<>
struct element_types<1> {
        typedef simplicial_element<1,0>    edge_type;
        typedef simplicial_element<1,1>    node_type;
};


} // namespace priv

template<typename T, size_t DIM>
using simplicial_mesh_storage = mesh_storage<T, DIM, simplicial_priv::element_types<DIM>>;

template<typename T, size_t DIM>
using simplicial_mesh = mesh<T, DIM, simplicial_mesh_storage<T, DIM>>;

template<typename T, size_t DIM>
std::array<simplicial_element<3,1>, 4>
faces(const simplicial_mesh<T, DIM>&, const simplicial_element<3,0>& vol)
{
    std::array<simplicial_element<3,1>, 4> ret;

    auto ptids = vol.point_ids();
    assert(ptids.size() == 4);

    ret[0] = simplicial_element<3,1>( { ptids[1], ptids[2], ptids[3] } );
    ret[1] = simplicial_element<3,1>( { ptids[0], ptids[2], ptids[3] } );
    ret[2] = simplicial_element<3,1>( { ptids[0], ptids[1], ptids[3] } );
    ret[3] = simplicial_element<3,1>( { ptids[0], ptids[1], ptids[2] } );

    return ret;
}

template<typename T, size_t DIM>
T
measure(const simplicial_mesh<T, DIM>& msh, const simplicial_element<3,0>& vol)
{
    auto pts = points(msh, vol);
    assert(pts.size() == 4);

    auto v0 = (pts[1] - pts[0]).to_vector();
    auto v1 = (pts[2] - pts[0]).to_vector();
    auto v2 = (pts[3] - pts[0]).to_vector();

    return abs( v0.dot(v1.cross(v2))/T(6) );
}

template<typename T, size_t DIM>
T
measure(const simplicial_mesh<T, DIM>& msh, const simplicial_element<3,1>& surf)
{
    auto pts = points(msh, surf);
    assert(pts.size() == 3);

    auto v0 = (pts[1] - pts[0]).to_vector();
    auto v1 = (pts[2] - pts[0]).to_vector();

    return v0.cross(v1).norm()/T(2);
}

} // namespace hho
