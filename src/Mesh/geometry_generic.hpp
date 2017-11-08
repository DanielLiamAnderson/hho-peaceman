#pragma once

#include "element_generic.hpp"
#include "gauss.hpp"

namespace hho {

const double EPSILON = 1e-8;

namespace generic_priv {

template<size_t DIM>
struct element_types
{
    static_assert(DIM > 0 && DIM <= 3, "element_types: CODIM must be less than DIM");
};

template<>
struct element_types<3> {
        typedef generic_element<3,0>    volume_type;
        typedef generic_element<3,1>    surface_type;
        typedef generic_element<3,2>    edge_type;
        typedef generic_element<3,3>    node_type;
};

template<>
struct element_types<2> {
        typedef generic_element<2,0>    surface_type;
        typedef generic_element<2,1>    edge_type;
        typedef generic_element<2,2>    node_type;
};

template<>
struct element_types<1> {
        typedef generic_element<1,0>    edge_type;
        typedef generic_element<1,1>    node_type;
};

} // namespace priv

template<typename T, size_t DIM>
using generic_mesh_storage = mesh_storage<T, DIM, generic_priv::element_types<DIM>>;

template<typename T, size_t DIM>
using generic_mesh = mesh<T, DIM, generic_mesh_storage<T, DIM>>;


template<typename T, size_t DIM>
size_t
number_of_faces(const generic_mesh<T,DIM>& msh, const typename generic_mesh<T,DIM>::cell& cl)
{
    return cl.subelement_size();
}

/* Return the offsets of all the faces of the specified cell */
template<typename T, size_t DIM>
std::vector<size_t>
global_face_offsets(const generic_mesh<T,DIM>& msh, const typename generic_mesh<T,DIM>::cell& cl)
{
    auto face_to_id = [&](const typename generic_mesh<T,DIM>::face::id_type& fid) -> size_t {
        return size_t( fid );
    };

    auto fcs = faces(msh, cl);
    std::vector<size_t> face_offsets( fcs.size() );

    std::transform(cl.subelement_id_begin(), cl.subelement_id_end(),
                   face_offsets.begin(), face_to_id);

    return face_offsets;
}

/* Return the offset of the specified face */
template<typename T, size_t DIM>
size_t
global_face_offset(const generic_mesh<T,DIM>& msh, const typename generic_mesh<T,DIM>::face& fc)
{
    return size_t( msh.lookup(fc) );
}

/* Return the offset of the specified face */
template<typename T, size_t DIM>
std::vector<size_t>
global_dofs_offset(const generic_mesh<T,DIM>& msh, const typename generic_mesh<T,DIM>::face& fc, size_t m_degree)
{
    auto num_face_dofs = num_dofs(fc, m_degree);
    std::vector<size_t> ret( num_face_dofs );
    for (size_t i = 0; i < num_face_dofs; i++)
        ret[i] = global_face_offset(msh, fc)*num_face_dofs + i;

    return ret;
}


/* Compute the measure of a 2-cell (= area) */
template<typename T>
T
measure(const generic_mesh<T,2>& msh, const typename generic_mesh<T,2>::cell& cl)
{
    auto pts = points(msh, cl);

    T acc{};
    for (size_t i = 1; i < pts.size() - 1; i++)
    {
        auto u = (pts.at(i) - pts.at(0)).to_vector();
        auto v = (pts.at(i+1) - pts.at(0)).to_vector();
        auto n = cross(u, v);
        acc += n.norm() / T(2);
    }

    return acc;
}

/* Compute the measure of a 2-face (= length) */
template<typename T>
T
measure(const generic_mesh<T,2>& msh, const typename generic_mesh<T,2>::face& fc)
{
    auto pts = points(msh, fc);
    assert(pts.size() == 2);
    return (pts[1] - pts[0]).to_vector().norm();
}




/* Compute the measure of a 1-cell (= area) */
template<typename T>
T
measure(const generic_mesh<T,1>& msh, const typename generic_mesh<T,1>::cell& cl)
{
    auto pts = points(msh, cl);
    assert(pts.size() == 2);
    return (pts[1] - pts[0]).to_vector().norm();
}

/* Compute the measure of a 1-face (= length) */
template<typename T>
T
measure(const generic_mesh<T,1>& msh, const typename generic_mesh<T,1>::face& fc)
{
    return T(1);
}

template<typename T>
T
diameter(const generic_mesh<T,2>& msh, const typename generic_mesh<T,2>::cell& cl)
{
    T c_meas = measure(msh, cl);
    T af_meas = T(0);
    auto fcs = faces(msh, cl);
    for (auto& f : fcs)
        af_meas += measure(msh, f);

    return c_meas/af_meas;
}

/* Compute an estimate of the mesh discretization step 'h' */
template<typename T>
T
mesh_h(const generic_mesh<T,2>& msh)
{
    T h{};
    for (auto itor = msh.cells_begin(); itor != msh.cells_end(); itor++)
    {
        auto cell = *itor;
        auto cell_measure = measure(msh, cell);
        //std::cout << cell_measure << std::endl;
        auto fcs = faces(msh, cell);
        T face_sum{};
        for (auto& f : fcs)
        {
            auto m = measure(msh, f);
            //std::cout << "    " << m << std::endl;
            face_sum += m;
        }
        h = std::max(h, cell_measure/face_sum);
    }

    return h;
}

/* Compute the barycenter of a 2-face */
template<typename T>
point<T,2>
barycenter(const generic_mesh<T,2>& msh, const typename generic_mesh<T,2>::face& fc)
{
    auto pts = points(msh, fc);
    assert(pts.size() == 2);
    auto bar = (pts[0] + pts[1]) / T(2);
    return bar;
}

template<typename T>
static_vector<T, 2>
normal(const generic_mesh<T,2>& msh, const typename generic_mesh<T,2>::face& fc)
{
    auto pts = points(msh, fc);
    assert(pts.size() == 2);

    auto u = pts[1] - pts[0];

    static_vector<T,2> ret;
    // XXX: wtf??
    ret(0) = -u.at(1);
    ret(1) = u.at(0);

    return ret/ret.norm();
}

template<typename T>
static_vector<T, 2>
normal(const generic_mesh<T,2>& msh, const typename generic_mesh<T,2>::face& fc,
       const point<T,2>& p)
{
    auto n = normal(msh, fc);
    auto pts = points(msh, fc);
    assert(pts.size() == 2);

    auto v = (pts[0] - p).to_vector();
    if ( v.dot(n) < T(0) )
        return -n;

    return n;
}

// Test whether the point c lies on the line segment ab
template<typename T>
bool contains_point(const point<T,2>& a, const point<T,2>& b, const point<T,2>& c) {
  return (std::abs((a - c).to_vector().norm() + (b - c).to_vector().norm() - (a - b).to_vector().norm()) < EPSILON);
}

// Test whether a 2D face (line segment) contains a given point
template<typename T>
bool contains_point(const generic_mesh<T,2>& mesh, const typename generic_mesh<T,2>::face& fc, const point<T,2>& x) {
  auto pts = points(mesh, fc);
  assert(pts.size() == 2);
  return contains_point(pts[0], pts[1], x);
}

// Test whether a 2D cell (polygon) contains a given point
// Points on the boundary are included
template<typename T>
bool contains_point(const generic_mesh<T,2>& mesh, const typename generic_mesh<T,2>::cell& cell, const point<T,2>& p) {
  const auto& poly = points(mesh, cell);
  size_t n = poly.size();
  bool c = false;
  
  // Point on boundary test
  for (size_t i = 0; i < n; i++)
    if (contains_point(poly[i], poly[(i + 1) % n], p)) return true;

  // Point inside test
  for (size_t i = 0, j = n - 1; i < n; j = i++)
    if (((poly[i].y() <= p.y() && p.y() < poly[j].y()) || (poly[j].y() <= p.y() && p.y() < poly[i].y())) &&
        (p.x() < (poly[j].x() - poly[i].x()) * (p.y() - poly[i].y()) / (poly[j].y() - poly[i].y()) + poly[i].x()))
      c = !c;

  return c;
}

/* Integrate on a cell of a generic 2D mesh (surface) */
template<typename T>
std::pair<std::vector<point<T,2>>, std::vector<T>>
integrate(const generic_mesh<T,2>& msh,
          const typename generic_mesh<T,2>::cell& cl,
          const typename generic_mesh<T,2>::face& fc,
          size_t doe)
{
    auto c_center       = barycenter(msh, cl);
    auto f_barycenter   = barycenter(msh, fc);
    auto f_measure      = measure(msh, fc);
    auto f_normal       = normal(msh, fc);
    auto d_cell_face    = fabs(f_normal.dot((f_barycenter - c_center).to_vector()));
    auto m_cell_face    = (d_cell_face * f_measure) / T(2);
    auto f_pts          = points(msh, fc);
    assert(f_pts.size() == 2);

    auto qr = dunavant_quadrature<T>(doe < 1 ? 1 : doe);

    auto c1 = f_pts[0] - c_center;
    auto c2 = f_pts[1] - c_center;

    static_matrix<T,2,2> M;
    M(0,0) = c1.at(0);
    M(0,1) = c2.at(0);
    M(1,0) = c1.at(1);
    M(1,1) = c2.at(1);

    auto points     = qr.first;
    auto weights    = qr.second;
    assert( points.size() == weights.size() );

    for (size_t i = 0; i < points.size(); i++)
    {
        auto newpt = M*points.at(i).to_vector() + c_center.to_vector();
        points.at(i) = point<T,2>({newpt(0), newpt(1)});
        weights.at(i) = m_cell_face * weights.at(i);
    }

    return std::make_pair(points, weights);
}

/* Integrate on a face of a generic 2D mesh (edge) */
template<typename T>
std::pair<std::vector<point<T,2>>, std::vector<T>>
integrate(const generic_mesh<T,2>& msh, const typename generic_mesh<T,2>::face& fc, size_t doe)
{
    auto pts = points(msh, fc);
    assert(pts.size() == 2);

    auto m = measure(msh, fc);

    auto qr = gauss_quadrature<T>(doe);

    std::vector<point<T,2>> points;
    points.reserve(qr.first.size());

    std::vector<T> weights;
    weights.reserve(qr.second.size());

    assert(points.size() == weights.size());

    for (int i = 0; i < qr.first.size(); i++)
    {
        points.push_back( pts[0] + qr.first(i) * (pts[1] - pts[0]) );
        weights.push_back(qr.second(i) * m);
    }

    return std::make_pair(points, weights);
}

} // namespace hho
