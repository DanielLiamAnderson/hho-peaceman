// file: integrator.hpp
// author: Daniel Anderson
//
// Computes integrals and norms of HHO grid functions.
// Particularly useful for estimating L2 error norms.
//
#pragma once

#include "common.hpp"

#include "hybrid_element.hpp"
#include "interpolation_operator.hpp"
#include "solver_base.hpp"

namespace hho {

template<size_t degree>
class integrator : public solver_base<degree> {
  
public:
  // An element encapsulating the local polynomial bases of a cell
  using element_type = typename solver_base<degree>::element_type;

  // The local operators that act on local degrees of freedom
  using interpolation_operator_type = interpolation_operator<element_type>;
  
  // Inherit base class members
  using solver_base<degree>::m_Th;                // the mesh
  using solver_base<degree>::m_gfn;               // the global face numbering
  using solver_base<degree>::m_dof_map;           // local -> global degree of freedom mapping
  using solver_base<degree>::m_local_elements;    // The local HHO elements
  
  // Inherit constructors
  using solver_base<degree>::solver_base;
  integrator(const solver_base<degree>& other) : solver_base<degree>(other) { }

  // Integrate the given HHO grid function globally over the mesh
  //
  // Arguments:
  //  X   the function to integrate
  scalar_type integrate(const dynamic_vector<scalar_type>& X) {
    if (cell_quadrature.size() == 0) {
      compute();
    }
    return cell_quadrature.dot(X.head(nb_cell_dofs));
  }  
  
  // Integrate a cellwise scalar function over the entire mesh
  //
  // Arguments:
  //  f   the function to integrate
  scalar_type integrate(const cellwise_scalar_function& f) {
    auto& Th = *m_Th;
    auto& local_elements = *m_local_elements;
    
    scalar_type res = 0.0;
    
    // For each cell T in Th
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = std::distance(Th.cells_begin(), it_T);
      auto FT = faces(Th, T);
      
      // Lookup local hybrid element
      element_type& E = local_elements[iT];
      
      // Assemble the cell quadrature
      // For each triangular subelement
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        auto iF_loc = std::distance(FT.begin(), it_F);
        const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);
        // For each quadrature node
        for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
          const auto& xQN = qr_PTF.first[iQN];
          const auto& wQN = qr_PTF.second[iQN];
          res += wQN * f(iT, xQN);
        }    // for iQN
      }      // for it_F
    }  // for it_T
    
    return res;
  }
  
  // Compute an L2 inner product of an HHO grid function and a meshwise scalar function
  //
  // Arguments:
  //  X   the grid function to integrate
  //  f   the function to integrate
  scalar_type l2_inner_product(const dynamic_vector<scalar_type>& X, const cellwise_scalar_function& f) {
    auto& Th = *m_Th;
    auto& local_elements = *m_local_elements;
    auto& dof_map = *m_dof_map;
    
    scalar_type res = 0.0;
    
    // For each cell T in Th
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = std::distance(Th.cells_begin(), it_T);
      auto FT = faces(Th, T);
      
      // Lookup local hybrid element
      element_type& E = local_elements[iT];
      
      // Local -> Global degrees of freedom indices
      const auto& dofs_T = dof_map[iT];
      
      // Assemble the cell quadrature
      // For each triangular subelement
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        auto iF_loc = std::distance(FT.begin(), it_F);
        const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);
        // For each quadrature node
        for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
          const auto& xQN = qr_PTF.first[iQN];
          const auto& wQN = qr_PTF.second[iQN];
          for (size_t i = 0; i < element_type::nb_local_cell_dofs; i++) {
            res += wQN * X(dofs_T(i)) * f(iT, xQN) * E.cell_functions_on_pyramid(iF_loc)(i, iQN);
          }  // for i
        }    // for iQN
      }      // for it_F
    }  // for it_T
    
    return res;
  }
  
  // Compute the L2 norm of the given HHO grid function
  //
  // Arguments:
  //  X   the grid function
  scalar_type l2_norm(const dynamic_vector<scalar_type>& X) {
    
    auto& Th = *m_Th;
    auto& local_elements = *m_local_elements;
    auto& dof_map = *m_dof_map;
    
    scalar_type norm = 0.0;
    
    // For each cell T in Th
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = std::distance(Th.cells_begin(), it_T);
      auto FT = faces(Th, T);
      
      // Lookup local hybrid element
      element_type& E = local_elements[iT];
      
      // Local -> Global degrees of freedom indices
      const auto& dofs_T = dof_map[iT];
      
      // Assemble the cell quadrature
      // For each triangular subelement
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        auto iF_loc = std::distance(FT.begin(), it_F);
        const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);
        // For each quadrature node
        for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
          const auto& wQN = qr_PTF.second[iQN];
          scalar_type u_xQN = 0.0;
          for (size_t i = 0; i < element_type::nb_local_cell_dofs; i++) {
            u_xQN += X(dofs_T(i)) * E.cell_functions_on_pyramid(iF_loc)(i, iQN);
          }  // for i
          norm += wQN * (u_xQN * u_xQN);
        }    // for iQN
      }      // for it_F
    }  // for it_T
    
    return pow(norm, 0.5);
  }
  
 private:
  // Pre-computes the integration coefficients for the polynomial basis over the mesh
  void compute() {
  
    auto& Th = *m_Th;
    auto& local_elements = *m_local_elements;
    auto& dof_map = *m_dof_map;
  
    // Count degrees of freedom
    nb_cell_dofs = Th.cells_size() * element_type::nb_local_cell_dofs;
    cell_quadrature = dynamic_vector<scalar_type>::Zero(nb_cell_dofs);
    
    // For each cell T in Th
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = std::distance(Th.cells_begin(), it_T);
      auto FT = faces(*Th, T);
      
      // Lookup local hybrid element
      element_type& E = local_elements[iT];
      
      // Local -> Global degrees of freedom indices
      const auto& dofs_T = dof_map[it_T];
      
      // Assemble the cell quadrature
      // For each triangular subelement
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        auto iF_loc = std::distance(FT.begin(), it_F);
        const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);
        // For each quadrature node
        for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
          const auto& wQN = qr_PTF.second[iQN];
          for (size_t i = 0; i < element_type::nb_local_cell_dofs; i++) {
            cell_quadrature(dofs_T(i)) += wQN * E.cell_functions_on_pyramid(iF_loc)(i, iQN);
          }  // for i
        }    // for iQN
      }      // for it_F
    }  // for it_T
  
  }
  
  // Member variables
  size_t nb_cell_dofs;
  dynamic_vector<scalar_type> cell_quadrature;
};

}  // namespace hho
