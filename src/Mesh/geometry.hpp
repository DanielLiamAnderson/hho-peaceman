/* HHO code - Geometry operations
 *
 * HISTORY:
 *  - 12/03/2016 (mc): File created
 *
 */

#pragma once

#include <algorithm>
#include <vector>

#include "mesh.hpp"
#include "util.h"

namespace hho {

template<typename Mesh>
std::vector<size_t>
global_face_offsets(const Mesh& msh, const typename Mesh::cell& cl)
{
    auto face_to_id = [&](const typename Mesh::face& fc) -> size_t {
        return size_t( msh.lookup(fc) );
    };

    auto fcs = faces(msh, cl);
    std::vector<size_t> face_offsets( fcs.size() );

    std::transform(fcs.begin(), fcs.end(), face_offsets.begin(), face_to_id);
    return face_offsets;
}

/* this must be moved in geometry_generic and specialized to generic_mesh */
template<typename Mesh>
std::vector<typename Mesh::face>
faces(const Mesh& msh, const typename Mesh::cell& cl)
{
    auto id_to_face = [&](const typename Mesh::face::id_type& id) -> typename Mesh::face {
        return *(msh.faces_begin() + size_t(id));
    };

    std::vector<typename Mesh::face> ret;
    ret.resize( cl.subelement_size() );

    std::transform(cl.subelement_id_begin(), cl.subelement_id_end(),
                   ret.begin(), id_to_face);

    return ret;
}

/* this must be moved in geometry_generic and specialized to generic_mesh */
template<typename Mesh>
std::vector<typename Mesh::face::id_type>
face_ids(const Mesh& msh, const typename Mesh::cell& cl)
{
    typedef typename Mesh::face::id_type        RetT;
    return std::vector<RetT>(cl.subelement_id_begin(), cl.subelement_id_end());
}

/* this is hho-specific and should not be here */
template<typename Mesh>
std::vector<size_t>
global_dofs_offsets(const Mesh& msh, const typename Mesh::cell& cl, size_t degree)
{
    auto num_face_dofs = num_dofs(typename Mesh::face(), degree);

    auto offsets = global_face_offsets(msh, cl);
    for (auto& off : offsets)
        off *= num_face_dofs;

    auto num_faces = offsets.size();

    std::vector<size_t> ret;
    ret.reserve(num_face_dofs * num_faces);

    for (size_t i = 0; i < num_faces; i++)
        for (size_t j = 0; j < num_face_dofs; j++)
            ret.push_back(offsets[i] + j);

    return ret;
}

template<typename Mesh, typename Element>
std::vector<typename Mesh::point_type>
points(const Mesh& msh, const Element& elem)
{
    auto ptids = elem.point_ids();

    auto ptid_to_point = [&](const point_identifier<Mesh::dimension>& pi) -> auto {
        auto itor = msh.points_begin();
        std::advance(itor, pi);
        return *itor;
    };

    std::vector<typename Mesh::point_type> pts(ptids.size());
    std::transform(ptids.begin(), ptids.end(), pts.begin(), ptid_to_point);

    return pts;
}

/* Compute the barycenter of a cell */
template<typename Mesh, typename Element>
point<typename Mesh::value_type, Mesh::dimension>
barycenter(const Mesh& msh, const Element& elm)
{
    auto pts = points(msh, elm);
    auto bar = std::accumulate(std::next(pts.begin()), pts.end(), pts.front());
    return bar / typename Mesh::value_type( pts.size() );
}

/* Determine the number of DoFs associated to the specified element */
template<template<size_t, size_t> class Element>
size_t
num_dofs(const Element<2,0>& elem, size_t degree)
{
    return binomial(degree+2, 2);
}

/*
template<template<size_t, size_t> class Element>
size_t
num_dofs<Element<2,0>>(size_t degree)
{
    return binomial(degree+2, 2);
}
*/
template<template<size_t, size_t> class Element>
size_t
num_all_dofs(const Element<2,0>& elem, size_t degree)
{
    size_t num_subelems = elem.subelement_size();
    return num_dofs(elem, degree) + num_subelems*(degree+1);
}

template<template<size_t, size_t> class Element>
size_t
num_dofs(const Element<2,1>& elem, size_t degree)
{
    return degree+1;
}
/*
template<template<size_t, size_t> class Element>
size_t
num_dofs<Element<2,1>>(size_t degree)
{
    return degree+1;
}
*/
#if 0
template<typename T>
[[deprecated("This function is not yet complete")]]
std::pair<std::vector<point<T,2>>, std::vector<T>>
integrate(const mesh<2,T>& msh, const face<2>& fc)
{
    std::vector<point<T,2>> pts;
    std::vector<T>          weights;

    auto face_meas = measure(msh, fc);
    auto face_pts = points(msh, fc);

    assert(face_pts.size() == 2); /* face<2> is an edge */

    /* Do integration here */
    (void)face_meas;

    return std::make_pair(pts, weights);
}
#endif


} // namespace hho
