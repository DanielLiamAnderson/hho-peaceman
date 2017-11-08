// file  : darcy_velocity_reconstruction.hpp
// author: Daniel Anderson
//
// Given the pressure solution, computes the reconstructed darcy velocity
// in the cells (volumetric terms) and on the faces (flux terms).
//
// Template Arugments:
//  degree    The degree of the HHO Scheme to use (0 <= degree <= 4)
#pragma once

#include <vector>

#include <Eigen/Sparse>

#include "hho/common.hpp"
#include "hho/hybrid_element.hpp"
#include "hho/solver_base.hpp"
#include "hho/variable_diffusion_operator.hpp"

namespace hho {

template<size_t degree>
class darcy_velocity_reconstruction : public solver_base<degree> {
 
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

  // The local operators that act on local degrees of freedom
  typedef variable_diffusion_operator<element_type> diffusion_operator_type;

  // Eigen matrix types and solvers
  typedef Eigen::Triplet<scalar_type> triplet_type;
  typedef Eigen::SparseMatrix<scalar_type> sparse_matrix_type;
  typedef Eigen::SparseLU<sparse_matrix_type> direct_solver_type;
  typedef Eigen::BiCGSTAB<sparse_matrix_type, Eigen::IncompleteLUT<scalar_type>> iterative_solver_type;

  // Inherit constructors
  using solver_base<degree>::solver_base;
  darcy_velocity_reconstruction(const solver_base<degree>& other) : solver_base<degree>(other) { }

  // Reconstructs the Darcy velocity and its fluxes from the given parameters
  //
  // Arguments:
  //  pTF       the solution to the HHO pressure equation
  //  kappa     the diffusion tensor as a function of the cell number and a point in the cell
  //
  std::pair<cellwise_vector_function, cellwise_flux_function> reconstruct(const dynamic_vector<scalar_type>& pTF,
    const cellwise_tensor_function kappa) {
  
    auto& Th = *m_Th;
    auto& local_elements = *m_local_elements;
    auto& dof_map = *m_dof_map;
    
    // Degree of freedom counts
    size_t nb_local_face_dofs = element_type::nb_local_face_dofs;
    
    // Create shared data
    auto flux_coeffs = std::make_shared<std::vector<std::vector<dynamic_vector<scalar_type>>>>(Th.cells_size(),
      std::vector<dynamic_vector<scalar_type>>());
    
    auto grad_pT = std::make_shared<std::vector<dynamic_vector<scalar_type>>>(Th.cells_size());
  
    // Do the reconstruction cell-wise
    // For each cell T in Th
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = distance(Th.cells_begin(), it_T);
      auto FT = faces(Th, T);
      size_t nb_local_dofs = element_type::nb_local_cell_dofs + FT.size() * nb_local_face_dofs;

      // Restrict kappa to the current cell
      auto kappa_T = [&](const point_type& x) { return kappa(iT, x); };

      // Create element
      element_type& E = local_elements[iT];

      // Compute local operators
      diffusion_operator_type a(kappa_T);
      auto a_T = a.compute(E).first;

      // Local -> Global degrees of freedom indices
      auto& dofs_T = dof_map[iT];
  
      // Extract local degrees of freedom from global
      dynamic_vector<scalar_type> pT = dynamic_vector<scalar_type>(nb_local_dofs);
      for (size_t i = 0; i < nb_local_dofs; i++) {
        pT(i) = pTF(dofs_T[i]);
      }
      
      // Compute the gradient reconstruction
      grad_pT->at(iT) = a.gradient_reconstruction() * pT;
      
      // Allocate flux coefficients
      flux_coeffs->at(iT).assign(FT.size(), dynamic_vector<scalar_type>(nb_local_face_dofs));
      
      // For each face of T
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        const auto iF_loc = std::distance(FT.begin(), it_F);
        size_t offset_F = element_type::nb_local_cell_dofs + iF_loc * nb_local_face_dofs;
        
        const auto& qr_F = E.quadrature_rule_on_face(iF_loc);
        
        // Assemble flux coefficients
        dynamic_vector<scalar_type> alpha = dynamic_vector<scalar_type>(nb_local_face_dofs);
        for (size_t i = 0; i < nb_local_face_dofs; i++) {
          alpha(i) = - pT.dot(a_T.col(offset_F + i));
        }
        
        // -----------------------------------------------------------------------------
        // Interface terms
        
        // Gram matrix
        dynamic_matrix<scalar_type> GFF = dynamic_matrix<scalar_type>::Zero(nb_local_face_dofs, nb_local_face_dofs);
        
        for (size_t iQN = 0; iQN < qr_F.first.size(); iQN++) {
          const auto& wQN = qr_F.second[iQN];
        
          // Build the Gram matrix
          for (size_t i = 0; i < nb_local_face_dofs; i++) {
            const auto& phi_i_iqn = E.face_functions_on_face(iF_loc)(i, iQN);
            for (size_t j = 0; j < nb_local_face_dofs; j++) {
              const auto& phi_j_iqn = E.face_functions_on_face(iF_loc)(j, iQN);
              GFF(i, j) += wQN * phi_i_iqn * phi_j_iqn;
            }  
          }
        }
     
        // Solve for the flux coefficients
        flux_coeffs->at(iT)[iF_loc] = GFF.ldlt().solve(alpha);
      }
    }
    
    // -----------------------------------------------------------------------------
    //              Assemble local reconstructed fluxes
    // -----------------------------------------------------------------------------
    
    // Assemble the reconstructed fluxes
    auto flux = [this, nb_local_face_dofs, flux_coeffs = std::move(flux_coeffs)](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      const auto& local_elements = *m_local_elements;
      const auto& E = local_elements[iT];
      
      scalar_type res = 0.0;
      for (size_t i = 0; i < nb_local_face_dofs; i++) {
        auto phi_i_x = E.face_basis(iF_loc).phi(i).phi(x);
        res += flux_coeffs->at(iT)[iF_loc](i) * phi_i_x;
      }
      return res;
    };
    
    // -----------------------------------------------------------------------------
    //              Assemble local reconstructed velocity
    // -----------------------------------------------------------------------------
    
    // Assemble the velocity
    auto velocity = [this, grad_pT = std::move(grad_pT), kappa = std::move(kappa)](const size_t iT, const point_type& x) {
      const auto& E = m_local_elements->at(iT);
      const auto& grad = grad_pT->at(iT);
      const auto& kappa_x = kappa(iT, x);
      
      vector_type G {0.0, 0.0};
      for (int i = 0; i < grad.size(); i++) {
        auto dphi_i_x = E.cell_basis().phi(E.cell_basis().degreeIndex(1) + i).dphi(x);
        G += grad(i) * dphi_i_x;
      }
      vector_type res = - kappa_x * G;
      return res;
    };
    
    return std::make_pair(velocity, flux);
  }
    
};

}  // namespace hho


