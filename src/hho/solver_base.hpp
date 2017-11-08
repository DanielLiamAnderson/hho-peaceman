// file  : solver_base.hpp
// author: Daniel Anderson
//
// Base class for generic HHO scheme solvers. Implements mesh loading
// and preprocessing (global face numbering), and local -> global
// degree of freedom mapping.
//
// Template Arugments:
//  degree    The degree of the HHO Scheme to use (0 <= degree <= 4)
#pragma once

#include "common.hpp"

#include <boost/filesystem.hpp>

namespace hho {

template<size_t degree>
class solver_base {
  
  // Constants
  int cell_doe = 3 * degree;    // the degree of exactness to use for polynomial quadrature
  int face_doe = 3 * degree;
  
 public:
  // Types
  // Element encapsulating the local polynomial bases of a cell
  using element_type = hybrid_element<mesh_type, degree, degree>;
 
  // Internal accessors to shared data
  const mesh_type& get_mesh() { return *m_Th; }
  const std::vector<element_type>& get_local_elements() { return *m_local_elements; }
  const std::map<mesh_type::face, size_t>& get_gfn() { return *m_gfn; }
  const std::vector<dynamic_vector<size_t>>& get_dof_map() { return *m_dof_map; }
  
  // Create a solver that will share the data used by an existing solver
  solver_base(const solver_base<degree>& other)
    : m_Th(other.m_Th),
      m_local_elements(other.m_local_elements),
      m_gfn(other.m_gfn),
      m_dof_map(other.m_dof_map) {
    // No further preprocessing required :)
  }
  
  // Create a solver that will load the given mesh
  solver_base(const std::string& mesh_filename) {
    load_mesh_from_file(mesh_filename);
    preprocess();
  }
 
  // Create a solver that will use the given mesh
  solver_base(const std::shared_ptr<mesh_type>& mesh) : m_Th(mesh) {
    preprocess();
  }
  
 protected:
  // Loads a mesh from the given file
  //
  // Arguments:
  //  mesh_filename  the filename of the mesh to load
  void load_mesh_from_file(const std::string& mesh_filename) {
    assert(boost::filesystem::exists(mesh_filename));
    fvca5_mesh_loader<double, 2> loader;
    loader.read_mesh(mesh_filename);
    m_Th = std::make_shared<mesh_type>();
    loader.populate_mesh(*m_Th);
  }
  
  // Preprocess the mesh, assign global indices to face degrees of freedom, compute
  // local elements and compute local -> global degree of freedom index maps
  void preprocess() {
    
    // Allocate shared data
    m_gfn = std::make_shared<std::map<mesh_type::face, size_t>>();
    m_local_elements = std::make_shared<std::vector<element_type>>();
    m_dof_map = std::make_shared<std::vector<dynamic_vector<size_t>>>();
    
    auto& Th = *m_Th;
    auto& gfn = *m_gfn;
    auto& local_elements = *m_local_elements;
    auto& dof_map = *m_dof_map;
  
    // ----------------------------------------------------------------------------------------
    // Assign global face numbering
    
    size_t index_F = 0;
    for (auto it_F = Th.internal_faces_begin(); it_F != Th.internal_faces_end(); it_F++) {
      gfn[*it_F] = index_F++;
    }  // for it_F
    for (auto it_F = Th.boundary_faces_begin(); it_F != Th.boundary_faces_end(); it_F++) {
      gfn[*it_F] = index_F++;
    }  // for it_F
    assert(index_F == Th.faces_size());
    
    // ----------------------------------------------------------------------------------------
    // Compute local elements
    
    // For each cell T in Th
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      local_elements.emplace_back(Th, T, cell_doe, face_doe);
    }
    
    // ----------------------------------------------------------------------------------------
    // Compute local -> global degree of freedom index maps
    
    // For each cell T in Th
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const size_t iT = std::distance<mesh_type::const_cell_iterator>(Th.cells_begin(), it_T);
      const size_t nb_local_dofs = element_type::nb_local_cell_dofs + T.subelement_size() * element_type::nb_local_face_dofs;

      size_t dof_index = 0;
      dof_map.emplace_back(nb_local_dofs);
      auto& res = dof_map[iT];

      // Count cell degrees of freedom
      for (size_t i = 0; i < element_type::nb_local_cell_dofs; i++) {
        res[dof_index++] = iT * element_type::nb_local_cell_dofs + i;
      }

      // Count face degrees of freedom
      auto FT = faces(Th, T);
      for (const auto& F : FT) {
        size_t offset = Th.cells_size() * element_type::nb_local_cell_dofs + gfn[F] * element_type::nb_local_face_dofs;
        for (size_t i = 0; i < element_type::nb_local_face_dofs; i++) {
          res[dof_index++] = offset + i;
        }  // for i
      }    // for F
    }
    
  }

  // ----------------------------------------------------------------------------------------
  // Shared Data
  
  std::shared_ptr<mesh_type> m_Th;                                // the mesh to be used by the solver
  std::shared_ptr<std::vector<element_type>> m_local_elements;    // The local HHO elements
  std::shared_ptr<std::map<mesh_type::face, size_t>> m_gfn;       // The global face numbering
  std::shared_ptr<std::vector<dynamic_vector<size_t>>> m_dof_map; // The local -> global degree of freedom map
};

}  // namespace hho


