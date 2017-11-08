/* HHO code - Mesh data structures
 *
 * HISTORY:
 *  - 05/02/2016 (mc): File created
 *
 */

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <cassert>
#include <iterator>

#include "ident.hpp"
#include "point.hpp"

namespace hho {

namespace priv {
    template<typename Iterator, typename Predicate>
    class filter_iterator
    {
        Predicate       predicate_;
        Iterator        itor_{}, end_{};

        void advance(void) { itor_++; }

        void find_next(void)
        {
            while ( itor_ != end_ )
            {
                if ( predicate_(*itor_) )
                    return;
                advance();
            }
        }

    public:
        using value_type = typename std::iterator_traits<Iterator>::value_type;
        using reference = typename std::iterator_traits<Iterator>::reference;

        filter_iterator() = default;

        filter_iterator(Predicate pred, Iterator itor, Iterator end)
            : predicate_(pred), itor_(itor), end_(end)
        {
            if (itor != end)
                find_next();
        }

        reference operator*() { return *itor_; }
        value_type *operator->() const { return &(*itor_); }

        filter_iterator& operator++() {
            if ( itor_ != end_ )
                advance();
            find_next();
            return *this;
        }

        filter_iterator operator++(int) {
            auto it = *this;
            ++(*this);
            return it;
        }

        bool operator==(const filter_iterator& other) { return (itor_ == other.itor_); }
        bool operator!=(const filter_iterator& other) { return (itor_ != other.itor_); }
    };

    template<typename mesh_type>
    class is_boundary_pred
    {
        const mesh_type&    msh_;
    public:
        is_boundary_pred(const mesh_type& msh) : msh_(msh) {}

        template<typename T>
        bool operator()(const T& elem) { return msh_.is_boundary(elem); }
    };

    template<typename mesh_type>
    class is_internal_pred
    {
        const mesh_type&    msh_;
    public:
        is_internal_pred(const mesh_type& msh) : msh_(msh) {}

        template<typename T>
        bool operator()(const T& elem) { return !msh_.is_boundary(elem); }
    };

} //namespace priv


template<typename T>
std::pair<bool, typename T::id_type>
find_element_id(const std::vector<T>& elements, const T& element)
{
    auto itor = std::lower_bound(elements.begin(), elements.end(), element);

    if (itor != elements.end() && !(element < *itor))
    {
        typename T::id_type id(std::distance(elements.begin(), itor));
        return std::make_pair(true, id);
    }

    return std::make_pair(false, typename T::id_type());
}

template<typename T, typename Iterator>
std::pair<bool, typename T::id_type>
find_element_id(const Iterator& begin, const Iterator& end, const T& element)
{
    auto itor = std::lower_bound(begin, end, element);

    if (itor != end && !(element < *itor))
    {
        typename T::id_type id(std::distance(begin, itor));
        return std::make_pair(true, id);
    }

    return std::make_pair(false, typename T::id_type());
}



/****************************************************************************/
namespace priv {

/* Mesh bones.
 *
 * @DIM     Space dimension
 * @T       Type that the mesh uses to represent points.
 */
template<typename T, size_t DIM, typename Storage>
class mesh_bones
{
    static_assert(DIM > 0 && DIM <= 3, "mesh: Allowed dimensions are 1, 2 and 3");

    std::shared_ptr<Storage>    m_storage;

public:
    mesh_bones() { m_storage = std::make_shared<Storage>(); }

    /* Return a shared_ptr to the backend storage. */
    std::shared_ptr<Storage>
    backend_storage(void)
    {
        return m_storage;
    }

    /* Return a shared_ptr to the backend storage. */
    const std::shared_ptr<Storage>
    backend_storage(void) const
    {
        return m_storage;
    }
};

/* Generic template for a mesh.
 *
 * This template has to be specialized for the 1D, 2D and 3D cases and it
 * represents the actual interface between the user and the mesh. It is in
 * `priv` and `mesh` inherits from `mesh_base` to provide an additional
 * decoupling layer.
 * The user should interact with the mesh in terms of cells and faces only.
 *
 * @DIM     Space dimension
 * @T       Type that the mesh uses to represent points.
 */
template<typename T, size_t DIM, typename Storage>
class mesh_base
{
    static_assert(DIM > 0 && DIM <= 3, "mesh: Allowed dimensions are 1, 2 and 3");
};



/* Template specialization for 3D meshes.
 *
 * @T       Type that the mesh uses to represent points.
 */
template<typename T, typename Storage>
class mesh_base<T,3,Storage> : public mesh_bones<T,3,Storage>
{
public:
    typedef typename Storage::volume_type                       volume_type;
    typedef typename Storage::surface_type                      surface_type;
    typedef typename Storage::edge_type                         edge_type;
    typedef typename Storage::node_type                         node_type;
    typedef T                                                   value_type;
    typedef typename Storage::point_type                        point_type;

    typedef volume_type                                         cell;
    typedef surface_type                                        face;
    const static size_t dimension = 3;

    /* cell iterators */
    typedef typename std::vector<volume_type>::iterator         cell_iterator;
    typedef typename std::vector<volume_type>::const_iterator   const_cell_iterator;

    cell_iterator           cells_begin() { return this->backend_storage()->volumes.begin(); }
    cell_iterator           cells_end()   { return this->backend_storage()->volumes.end(); }
    const_cell_iterator     cells_begin() const { return this->backend_storage()->volumes.begin(); }
    const_cell_iterator     cells_end()   const { return this->backend_storage()->volumes.end(); }

    /* face iterators */
    typedef typename std::vector<surface_type>::iterator        face_iterator;
    typedef typename std::vector<surface_type>::const_iterator  const_face_iterator;

    face_iterator           faces_begin() { return this->backend_storage()->surfaces.begin(); }
    face_iterator           faces_end()   { return this->backend_storage()->surfaces.end(); }
    const_face_iterator     faces_begin() const { return this->backend_storage()->surfaces.begin(); }
    const_face_iterator     faces_end()   const { return this->backend_storage()->surfaces.end(); }

    size_t  cells_size() const { return this->backend_storage()->volumes.size(); }
    size_t  faces_size() const { return this->backend_storage()->surfaces.size(); }

    bool is_boundary(typename face::id_type id) const
    {
        return this->backend_storage()->boundary_surfaces.at(id);
    }

    bool is_boundary(const face& f) const
    {
        auto e = find_element_id(this->backend_storage()->surfaces, f);
        if (e.first == false)
            throw std::invalid_argument("Cell not found");

        return this->backend_storage()->boundary_surfaces.at(e.second);
    }

    bool is_boundary(const face_iterator& itor) const
    {
        auto ofs = std::distance(faces_begin(), itor);
        return this->backend_storage()->boundary_surfaces.at(ofs);
    }

    size_t  boundary_faces_size() const
    {
        return std::count(this->backend_storage()->boundary_surfaces.begin(),
                          this->backend_storage()->boundary_surfaces.end(),
                          true);
    }

    size_t  internal_faces_size() const
    {
        return std::count(this->backend_storage()->boundary_surfaces.begin(),
                          this->backend_storage()->boundary_surfaces.end(),
                          false);
    }
};



/* Template specialization for 2D meshes.
 *
 * @T       Type that the mesh uses to represent points.
 */
template<typename T, typename Storage>
class mesh_base<T,2,Storage> : public mesh_bones<T,2,Storage>
{
public:
    typedef typename Storage::surface_type                      surface_type;
    typedef typename Storage::edge_type                         edge_type;
    typedef typename Storage::node_type                         node_type;
    typedef T                                                   value_type;
    typedef typename Storage::point_type                        point_type;

    typedef surface_type                                        cell;
    typedef edge_type                                           face;
    const static size_t dimension = 2;

    /* cell iterators */
    typedef typename std::vector<surface_type>::iterator        cell_iterator;
    typedef typename std::vector<surface_type>::const_iterator  const_cell_iterator;

    cell_iterator           cells_begin() { return this->backend_storage()->surfaces.begin(); }
    cell_iterator           cells_end()   { return this->backend_storage()->surfaces.end(); }
    const_cell_iterator     cells_begin() const { return this->backend_storage()->surfaces.begin(); }
    const_cell_iterator     cells_end()   const { return this->backend_storage()->surfaces.end(); }

    /* face iterators */
    typedef typename std::vector<edge_type>::iterator           face_iterator;
    typedef typename std::vector<edge_type>::const_iterator     const_face_iterator;

    face_iterator           faces_begin() { return this->backend_storage()->edges.begin(); }
    face_iterator           faces_end()   { return this->backend_storage()->edges.end(); }
    const_face_iterator     faces_begin() const { return this->backend_storage()->edges.begin(); }
    const_face_iterator     faces_end()   const { return this->backend_storage()->edges.end(); }

    size_t  cells_size() const { return this->backend_storage()->surfaces.size(); }
    size_t  faces_size() const { return this->backend_storage()->edges.size(); }

    bool is_boundary(typename face::id_type id) const
    {
        return this->backend_storage()->boundary_edges.at(id);
    }

    bool is_boundary(const face& f) const
    {
        auto e = find_element_id(this->backend_storage()->edges, f);
        if (e.first == false)
            throw std::invalid_argument("Cell not found");

        return this->backend_storage()->boundary_edges.at(e.second);
    }

    bool is_boundary(const face_iterator& itor) const
    {
        auto ofs = std::distance(faces_begin(), itor);
        return this->backend_storage()->boundary_edges.at(ofs);
    }

    size_t  boundary_faces_size() const
    {
        return std::count(this->backend_storage()->boundary_edges.begin(),
                          this->backend_storage()->boundary_edges.end(),
                          true);
    }

    size_t  internal_faces_size() const
    {
        return std::count(this->backend_storage()->boundary_edges.begin(),
                          this->backend_storage()->boundary_edges.end(),
                          false);
    }
};

/* mesh base class defining the data arrays for the 1D case */
template<typename T, typename Storage>
class mesh_base<T,1,Storage> : public mesh_bones<T,1,Storage>
{
public:
    typedef typename Storage::edge_type                     edge_type;
    typedef typename Storage::node_type                     node_type;
    typedef T                                               value_type;
    typedef typename Storage::point_type                    point_type;

    typedef edge_type                                       cell;
    typedef node_type                                       face;
    const static size_t dimension = 1;

    /* cell iterators */
    typedef typename std::vector<edge_type>::iterator       cell_iterator;
    typedef typename std::vector<edge_type>::const_iterator const_cell_iterator;

    cell_iterator           cells_begin() { return this->backend_storage()->edges.begin(); }
    cell_iterator           cells_end()   { return this->backend_storage()->edges.end(); }
    const_cell_iterator     cells_begin() const { return this->backend_storage()->edges.begin(); }
    const_cell_iterator     cells_end()   const { return this->backend_storage()->edges.end(); }

    /* face iterators */
    typedef typename std::vector<node_type>::iterator       face_iterator;
    typedef typename std::vector<node_type>::const_iterator const_face_iterator;

    face_iterator           faces_begin() { return this->backend_storage()->nodes.begin(); }
    face_iterator           faces_end()   { return this->backend_storage()->nodes.end(); }
    const_face_iterator     faces_begin() const { return this->backend_storage()->nodes.begin(); }
    const_face_iterator     faces_end()   const { return this->backend_storage()->nodes.end(); }

    size_t  cells_size() const { return this->backend_storage()->edges.size(); }
    size_t  faces_size() const { return this->backend_storage()->nodes.size(); }

    bool is_boundary(typename face::id_type id) const
    {
        return this->backend_storage()->boundary_nodes.at(id);
    }

    bool is_boundary(const face& f) const
    {
        auto e = find_element_id(this->backend_storage()->nodes, f);
        if (e.first == false)
            throw std::invalid_argument("Cell not found");

        return this->backend_storage()->boundary_nodes.at(e.second);
    }

    bool is_boundary(const face_iterator& itor) const
    {
        auto ofs = std::distance(faces_begin(), itor);
        return this->backend_storage()->boundary_nodes.at(ofs);
    }

    size_t  boundary_faces_size() const
    {
        return std::count(this->backend_storage()->boundary_nodes.begin(),
                          this->backend_storage()->boundary_nodes.end(),
                          true);
    }

    size_t  internal_faces_size() const
    {
        return std::count(this->backend_storage()->boundary_nodes.begin(),
                          this->backend_storage()->boundary_nodes.end(),
                          false);
    }
};

} // namespace priv

template<typename T, size_t DIM, typename Storage>
class mesh : public priv::mesh_base<T,DIM,Storage>
{
public:
    static const size_t dimension = DIM;
    typedef T scalar_type;

    typedef typename priv::mesh_base<T, DIM, Storage>::point_type   point_type;
    typedef typename priv::mesh_base<T, DIM, Storage>::cell         cell;
    typedef typename priv::mesh_base<T, DIM, Storage>::face         face;

    /* point iterators */
    typedef typename std::vector<point_type>::iterator              point_iterator;
    typedef typename std::vector<point_type>::const_iterator        const_point_iterator;

    point_iterator          points_begin() { return this->backend_storage()->points.begin(); }
    point_iterator          points_end()   { return this->backend_storage()->points.end(); }
    const_point_iterator    points_begin() const { return this->backend_storage()->points.begin(); }
    const_point_iterator    points_end()   const { return this->backend_storage()->points.end(); }

    size_t  points_size() const { return this->backend_storage()->points.size(); }

    typedef priv::filter_iterator<typename mesh::face_iterator,
                                  priv::is_boundary_pred<mesh>>
                                  boundary_face_iterator;

    typedef priv::filter_iterator<typename mesh::face_iterator,
                                  priv::is_internal_pred<mesh>>
                                  internal_face_iterator;

    boundary_face_iterator  boundary_faces_begin()
    {
        typedef priv::is_boundary_pred<mesh> ibp;
        return boundary_face_iterator(ibp(*this), this->faces_begin(), this->faces_end());
    }

    boundary_face_iterator  boundary_faces_end()
    {
        typedef priv::is_boundary_pred<mesh> ibp;
        return boundary_face_iterator(ibp(*this), this->faces_end(), this->faces_end());
    }

    internal_face_iterator  internal_faces_begin()
    {
        typedef priv::is_internal_pred<mesh> iip;
        return internal_face_iterator(iip(*this), this->faces_begin(), this->faces_end());
    }

    internal_face_iterator  internal_faces_end()
    {
        typedef priv::is_internal_pred<mesh> iip;
        return internal_face_iterator(iip(*this), this->faces_end(), this->faces_end());
    }

    /* Apply a transformation to the mesh. Transform should be a functor or
     * a lambda function of type
     *      mesh_type::point_type -> mesh_type::point_type
     */
    template<typename Transform>
    void transform(const Transform& tr)
    {
        std::transform(points_begin(), points_end(), points_begin(), tr);
    }

    /* Returns the numerial ID of a cell. */
    typename cell::id_type lookup(const cell& cl) const
    {
        auto ci = find_element_id(this->cells_begin(), this->cells_end(), cl);
        if (!ci.first)
            throw std::invalid_argument("Cell not present in mesh");

        return ci.second;
    }

    /* Returns the numerial ID of a face. */
    typename face::id_type lookup(const face& fc) const
    {
        auto fi = find_element_id(this->faces_begin(), this->faces_end(), fc);
        if (!fi.first)
            throw std::invalid_argument("Face not present in mesh");

        return fi.second;
    }

    /* Th->maximumNumberOfFaces() */
    [[deprecated("subelement_size works only on generic_element")]]
    size_t  max_faces_per_element(void) const
    {
        size_t mfpe = 0;
        for (auto itor = this->cells_begin(); itor != this->cells_end(); itor++)
        {
            auto cell = *itor;
            mfpe = std::max(mfpe, cell.subelement_size());
        }

        return mfpe;
    }
};

template<typename T, size_t DIM, typename Storage>
typename mesh<T, DIM, Storage>::cell_iterator
begin(mesh<T, DIM, Storage>& msh)
{
    return msh.cells_begin();
}

template<typename T, size_t DIM, typename Storage>
typename mesh<T, DIM, Storage>::cell_iterator
end(mesh<T, DIM, Storage>& msh)
{
    return msh.cells_end();
}

template<typename T, size_t DIM, typename Storage>
typename mesh<T, DIM, Storage>::const_cell_iterator
begin(const mesh<T, DIM, Storage>& msh)
{
    return msh.cells_begin();
}

template<typename T, size_t DIM, typename Storage>
typename mesh<T, DIM, Storage>::const_cell_iterator
end(const mesh<T, DIM, Storage>& msh)
{
    return msh.cells_end();
}

} // namespace hho
