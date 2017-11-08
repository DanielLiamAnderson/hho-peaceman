// file  : function_reconstruction.hpp
// author: Daniel Anderson
//
// Given an HHO grid function, reconstructs the volmetric part of the 
// function cell-wise.
//
// Template Arugments:
//  degree    The degree of the HHO Scheme to use (0 <= degree <= 4)
#pragma once

#include "hho/common.hpp"
#include "hho/hybrid_element.hpp"
#include "hho/solver_base.hpp"

namespace hho {

template<size_t degree>
class function_reconstruction : public solver_base<degree> {
 
  // Inherit base class members
  using solver_base<degree>::m_Th;              // the mesh
  using solver_base<degree>::m_gfn;             // the global face numbering
  using solver_base<degree>::m_dof_map;         // local -> global degree of freedom mapping
  using solver_base<degree>::m_local_elements;  // the local hybrid elements
  
  //------------------------------------------------------------------------------
  // Internal Typedefs
 public:
  // An element encapsulating the local polynomial bases of a cell
  typedef typename solver_base<degree>::element_type element_type;

  // Inherit constructors
  using solver_base<degree>::solver_base;
  function_reconstruction(const solver_base<degree>& other) : solver_base<degree>(other) { }

  // Reconstructs the given function. Data is moved to avoid incuring a copy.
  //
  // Arguments:
  //  XTF       the HHO grid function.
  //
  cellwise_scalar_function reconstruct(dynamic_vector<scalar_type>&& XTF) {
    return reconstruct(std::make_shared<dynamic_vector<scalar_type>>(std::move(XTF)));
  }
  
  // Reconstructs the given function. Prefer to move XTF or use a shared_ptr to XTF
  // to avoid incuring a copy of the data.
  //
  // Arguments:
  //  XTF       the HHO grid function.
  //
  cellwise_scalar_function reconstruct(const dynamic_vector<scalar_type>& XTF) {
    return reconstruct(std::make_shared<dynamic_vector<scalar_type>>(XTF));
  }

  // Reconstructs the given function from shared data. This is the recommended way
  // to use this function.
  //
  // Arguments:
  //  Xh       a pointer to the HHO grid function. The reconstruction will share ownership
  //
  cellwise_scalar_function reconstruct(std::shared_ptr<dynamic_vector<scalar_type>> XTF) {

    // ---------------------------------------------------------------
    //                    Assemble the function
    // ---------------------------------------------------------------
    
    // Assemble the reconstructed function
    auto reconstructed_function = [this, Xh = std::move(XTF)](const size_t iT, const point_type& x) {
      const element_type& E = m_local_elements->at(iT);
      auto& dofs_T = m_dof_map->at(iT);
      scalar_type res = 0.0;
      for (size_t i = 0; i < E.nb_cell_dofs(); i++) {
        auto phi_i_x = E.cell_basis().phi(i).phi(x);
        res += (*Xh)(dofs_T(i)) * phi_i_x;
      }
      return res;
    };
    
    return reconstructed_function;
    
  }
};

}  // namespace hho


