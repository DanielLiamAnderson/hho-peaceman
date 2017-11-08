// file: global_interpolator.hpp
// author: Daniel Anderson
//
// Constructs HHO polynomial interpolants of given functions
// onto a given mesh.
//
#pragma once

#include "common.hpp"

#include "hybrid_element.hpp"
#include "interpolation_operator.hpp"
#include "solver_base.hpp"

namespace hho {

template<size_t degree>
class global_interpolator : public solver_base<degree> {
  
public:
  // An element encapsulating the local polynomial bases of a cell
  using element_type = typename solver_base<degree>::element_type;

  // The local operators that act on local degrees of freedom
  using interpolation_operator_type = interpolation_operator<element_type>;
  
  // Inherit base class members
  using solver_base<degree>::m_Th;                // the mesh
  using solver_base<degree>::m_gfn;               // the global face numbering
  using solver_base<degree>::m_dof_map;           // local -> global degree of freedom mapping
  using solver_base<degree>::m_local_elements;    // the local HHO elements
  
  // Inherit constructors
  using solver_base<degree>::solver_base;
  global_interpolator(const solver_base<degree>& other) : solver_base<degree>(other) { }
  
  // Interpolate the given function globally over the mesh
  //
  // Arguments:
  //  u   the function to interpolate
  dynamic_vector<scalar_type> global_interpolant(const cellwise_scalar_function& u) {
    auto& Th = *m_Th;
    auto& gfn = *m_gfn;
    auto& local_elements = *m_local_elements;
  
    // Count degrees of freedom
    size_t nb_internal_face_dofs = Th.internal_faces_size() * element_type::nb_local_face_dofs;
    size_t nb_cell_dofs = Th.cells_size() * element_type::nb_local_cell_dofs;
    size_t nb_boundary_dofs = Th.boundary_faces_size() * element_type::nb_local_face_dofs;
    size_t nb_tot_dofs = nb_cell_dofs + nb_internal_face_dofs + nb_boundary_dofs;
    
    dynamic_vector<scalar_type> XTF = dynamic_vector<scalar_type>::Zero(nb_tot_dofs);
    
    // For each cell T in Th
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      size_t iT = std::distance(Th.cells_begin(), it_T);
      auto FT = faces(Th, T);
      
      // Create element
      element_type& E = local_elements[iT];
      
      // Restrict function to this cell
      auto uT = [&](const point_type& x) { return u(iT, x); };
      
      // Interpolate the exact solution
      interpolation_operator_type interp(uT);
      auto interp_T = interp.compute(E);

      dynamic_vector<scalar_type> UT = interp_T.first.ldlt().solve(interp_T.second);

      // Assemble interpolate vector
      size_t offset_T = iT * element_type::nb_local_cell_dofs;
      XTF.segment<element_type::nb_local_cell_dofs>(offset_T) = UT.head<element_type::nb_local_cell_dofs>();
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        auto F = *it_F;
        size_t offset_F = gfn[F] * element_type::nb_local_face_dofs;
        auto iF_loc = std::distance(FT.begin(), it_F);
        size_t offset_F_loc = element_type::nb_local_cell_dofs + iF_loc * element_type::nb_local_face_dofs;
        XTF.segment<element_type::nb_local_face_dofs>(nb_cell_dofs + offset_F) =
            UT.segment<element_type::nb_local_face_dofs>(offset_F_loc);
      }  // for it_F
    }  // for it_T
    
    return XTF;
  }
  
  // Interpolate the given function globally over the mesh
  //
  // Arguments:
  //  u   the function to interpolate
  dynamic_vector<scalar_type> global_interpolant(const scalar_function_of_space& u) {
    auto uT = [&](const size_t iT, const point_type& x) { return u(x); };
    return global_interpolant(uT);
  }  
};

}  // namespace hho
