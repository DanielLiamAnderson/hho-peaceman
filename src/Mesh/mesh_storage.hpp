#pragma once

#include <vector>

#include "point.hpp"

namespace hho {

/* Generic template for mesh_storage.
 *
 * This template has to be specialized for the 1D, 2D and 3D cases.
 * The function of mesh_storage is to decouple the low-level details of the
 * storage of the mesh elements (points, nodes, edges...) from the actual view
 * on the mesh. mesh_storage is part of the internal representation of the
 * mesh, for this reason is in the `priv` namespace. Internal representation
 * could change, breaking the user code, so all accesses have to be proxied by
 * `mesh_base`.
 * However, in some situations, accessing the internal representation can be
 * useful if not essential. This is the case for example of the mesh loaders.
 *
 * @DIM     Space dimension
 * @T       Type that the mesh uses to represent points.
 */
template<typename T, size_t DIM, typename ElementTypes>
class mesh_storage
{
    static_assert(DIM > 0 && DIM <= 3, "mesh: Allowed dimensions are 1, 2 and 3");
};

/* Template specialization of mesh_storage for 3D meshes.
 *
 * @T       Type that the mesh uses to represent points.
 */
template<typename T, typename ElementTypes>
struct mesh_storage<T, 3, ElementTypes>
{
    typedef typename ElementTypes::volume_type  volume_type;
    typedef typename ElementTypes::surface_type surface_type;
    typedef typename ElementTypes::edge_type    edge_type;
    typedef typename ElementTypes::node_type    node_type;
    typedef T                                   value_type;
    typedef point<T,3>                          point_type;

    std::vector<volume_type>                    volumes;
    std::vector<surface_type>                   surfaces;
    std::vector<edge_type>                      edges;
    std::vector<node_type>                      nodes;
    std::vector<point<T,3>>                     points;

    std::vector<bool>                           boundary_surfaces;

    void statistics(void) const
    {
        std::cout << "This is a storage for a 3D mesh" << std::endl;
        std::cout << "Points: " << points.size() << std::endl;
        std::cout << "Nodes: " << nodes.size() << std::endl;
        std::cout << "Edges: " << edges.size() << std::endl;
        std::cout << "Surfaces: " << surfaces.size() << std::endl;
        std::cout << "Volumes: " << volumes.size() << std::endl;
    }
};

/* Template specialization of mesh_storage for 2D meshes.
 *
 * @T       Type that the mesh uses to represent points.
 */
template<typename T, typename ElementTypes>
struct mesh_storage<T, 2, ElementTypes>
{
    typedef typename ElementTypes::surface_type surface_type;
    typedef typename ElementTypes::edge_type    edge_type;
    typedef typename ElementTypes::node_type    node_type;
    typedef T                                   value_type;
    typedef point<T,2>                          point_type;

    std::vector<surface_type>                   surfaces;
    std::vector<edge_type>                      edges;
    std::vector<node_type>                      nodes;
    std::vector<point<T,2>>                     points;

    std::vector<bool>                           boundary_edges;

    void statistics(void) const
    {
        std::cout << "This is a storage for a 2D mesh" << std::endl;
        std::cout << "Points: " << points.size() << std::endl;
        std::cout << "Nodes: " << nodes.size() << std::endl;
        std::cout << "Edges: " << edges.size() << std::endl;
        std::cout << "Surfaces: " << surfaces.size() << std::endl;
    }
};

/* Template specialization of mesh_storage for 1D meshes.
 *
 * @T       Type that the mesh uses to represent points.
 */
 template<typename T, typename ElementTypes>
 struct mesh_storage<T, 1, ElementTypes>
 {
     typedef typename ElementTypes::edge_type    edge_type;
     typedef typename ElementTypes::node_type    node_type;
     typedef T                                   value_type;
     typedef point<T,1>                          point_type;

     std::vector<edge_type>                      edges;
     std::vector<node_type>                      nodes;
     std::vector<point<T,1>>                     points;

     std::vector<bool>                           boundary_nodes;

     void statistics(void) const
     {
         std::cout << "This is a storage for a 1D mesh" << std::endl;
         std::cout << "Points: " << points.size() << std::endl;
         std::cout << "Nodes: " << nodes.size() << std::endl;
         std::cout << "Edges: " << edges.size() << std::endl;
     }
 };

} // namespace hho
