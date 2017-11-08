// file  : adr_solver.hpp
// author: Daniel Anderson
//
// Solver for the advection-diffusion-reaction equation
// with Neumann boundary conditions.
//
// Template Arugments:
//  degree    The degree of the HHO Scheme to use (0 <= degree <= 4)
#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include "common.hpp"

#include "advection_reaction_operator.hpp"
#include "variable_diffusion_operator.hpp"
#include "hybrid_element.hpp"
#include "load_operator.hpp"
#include "solver_base.hpp"

namespace hho {

template<size_t degree>
class adr_solver : public solver_base<degree> {
 
  // Inherit base class members
  using solver_base<degree>::m_Th;                // the mesh
  using solver_base<degree>::m_gfn;               // the global face numbering
  using solver_base<degree>::m_dof_map;           // local -> global degree of freedom mapping
  using solver_base<degree>::m_local_elements;    // the local HHO elements
  
  //------------------------------------------------------------------------------
  // Internal types
 public:
  // Inherit the local element type
  using element_type = typename solver_base<degree>::element_type;
  
  // The local operators that act on local degrees of freedom
  using diffusion_operator_type = variable_diffusion_operator<element_type>;
  using advection_operator_type = advection_reaction_operator<element_type>;
  using load_operator_type = load_operator<element_type>;

  // Eigen matrix types and solvers
  using triplet_type = Eigen::Triplet<scalar_type>;
  using sparse_matrix_type = Eigen::SparseMatrix<scalar_type>;
  using direct_solver_type = Eigen::SparseLU<sparse_matrix_type>;
  using iterative_solver_type = Eigen::BiCGSTAB<sparse_matrix_type, Eigen::IncompleteLUT<scalar_type>>;

  // Inherit constructors
  using solver_base<degree>::solver_base;
  adr_solver(const solver_base<degree>& other) : solver_base<degree>(other) { }

  // Solves the Advection-Diffusion-Reaction equation using the given data and returns the HHO solution
  //
  // Arguments:
  //  source    the source term f
  //  boundary  the value of the directional derrivative of the solution at the given point, in the given direction
  //  diffusion_tensor  the diffusion tensor as a function of space 
  //  velocity  the velocity vector as a function of space
  //  flux      the flux of the velocity in a given direction as a function of space
  //  reaction  the reaction terms as a function of space
  //
  dynamic_vector<scalar_type> solve(const cellwise_scalar_function& source, const cellwise_flux_function& boundary,
    const cellwise_tensor_function& diffusion_tensor, const cellwise_vector_function& velocity,
    const cellwise_flux_function& flux, const cellwise_scalar_function& reaction) {
    
      auto& Th = *m_Th;
      auto& gfn = *m_gfn;
      auto& dof_map = *m_dof_map;
      auto& local_elements = *m_local_elements;
    
      //------------------------------------------------------------------------------
      // Global matrix and vector

      // Degree of freedom counts
      size_t nb_internal_face_dofs = Th.internal_faces_size() * element_type::nb_local_face_dofs;
      size_t nb_cell_dofs = Th.cells_size() * element_type::nb_local_cell_dofs;
      size_t nb_boundary_dofs = Th.boundary_faces_size() * element_type::nb_local_face_dofs;
      size_t nb_face_dofs = nb_internal_face_dofs + nb_boundary_dofs;
      size_t nb_tot_dofs = nb_cell_dofs + nb_internal_face_dofs + nb_boundary_dofs;

      sparse_matrix_type A(nb_face_dofs, nb_face_dofs);
      sparse_matrix_type A_sc(nb_cell_dofs, nb_face_dofs);
      
      dynamic_vector<scalar_type> B = dynamic_vector<scalar_type>::Zero(nb_face_dofs);
      dynamic_vector<scalar_type> B_sc = dynamic_vector<scalar_type>::Zero(nb_cell_dofs);
    
      //------------------------------------------------------------------------------
      // Assemble local contributions

      std::vector<triplet_type> triplets;
      std::vector<triplet_type> triplets_sc;

      // For each cell T in Th
      for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
        const auto& T = *it_T;
        const auto iT = std::distance(Th.cells_begin(), it_T);
        auto xT = barycenter(Th, T);
        auto FT = faces(Th, T);

        // Get local element
        element_type& E = local_elements[iT];

        // Create local restrictions of the data to the cell T
        auto kappa_T = [&](const point_type& x) { return diffusion_tensor(iT, x); };
        auto f_T = [&](const point_type& x) { return source(iT, x); };
        auto beta_T = [&](const point_type& x) { return velocity(iT, x); };
        auto mu_T = [&](const point_type& x) { return reaction(iT, x); };
        auto beta_TF = [&](const size_t iF_loc, const point_type& x, const vector_type& nTF) {
          return flux(iT, iF_loc, x, nTF);
        };

        // Compute local operators
        diffusion_operator_type ad(kappa_T);
        advection_operator_type aar(beta_T, beta_TF, mu_T, kappa_T);
        load_operator_type l(f_T);

        auto a_kappa_T = ad.compute(E).first;
        auto a_beta_mu = aar.compute(E).first;
        dynamic_matrix<scalar_type> aT = a_kappa_T + a_beta_mu;
        auto bT = l.compute(E).second;

        // Perform static condensation
        dynamic_matrix<scalar_type> ATT = aT.topLeftCorner(element_type::nb_local_cell_dofs, element_type::nb_local_cell_dofs);
        dynamic_matrix<scalar_type> ATF = aT.topRightCorner(E.nb_cell_dofs(), E.nb_face_dofs());
        dynamic_matrix<scalar_type> AFT = aT.bottomLeftCorner(E.nb_face_dofs(), E.nb_cell_dofs());
        dynamic_matrix<scalar_type> AFF = aT.bottomRightCorner(E.nb_face_dofs(), E.nb_face_dofs());
        
        Eigen::PartialPivLU<dynamic_matrix<scalar_type>> invATT;
        invATT.compute(ATT);
        
        dynamic_matrix<scalar_type> invATT_ATF = invATT.solve(ATF);
        dynamic_vector<scalar_type> invATT_bT = invATT.solve(bT);
        AFF -= AFT * invATT_ATF;
        dynamic_vector<scalar_type> bF = - AFT * invATT_bT;

        // Local -> Global degrees of freedom indices
        const auto& dofs_T = dof_map[iT];

        // Assemble local contribution into global matrix
        constexpr size_t cell_offset = element_type::nb_local_cell_dofs;
        for (size_t i = 0; i < E.nb_face_dofs(); i++) {
          for (size_t j = 0; j < E.nb_face_dofs(); j++) {
            triplets.emplace_back(
              dofs_T(i + cell_offset) - nb_cell_dofs,
              dofs_T(j + cell_offset) - nb_cell_dofs,
              AFF(i, j));
          }  // for j
          B(dofs_T(i + cell_offset) - nb_cell_dofs) += bF(i);
        }    // for i
        
        // Assemble static condensation operator
        size_t offset_T = iT * cell_offset;
        B_sc.segment<cell_offset>(offset_T) = invATT_bT;
        for (size_t i = 0; i < cell_offset; i++) {
          for (size_t j = 0; j < E.nb_face_dofs(); j++) {
            triplets_sc.emplace_back(
              offset_T + i,
              dofs_T(j + cell_offset) - nb_cell_dofs,
              invATT_ATF(i, j));
          }
        }

        // Assemble the boundary source term for neumann conditions
        for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
          const auto& F = *it_F;
          if (Th.is_boundary(F)) {
            auto iF_loc = std::distance(FT.begin(), it_F);
            const auto& qr_F = E.quadrature_rule_on_face(iF_loc);
            auto nTF = normal(Th, F, xT);
            // For each quadrature node
            for (size_t iQN = 0; iQN < qr_F.first.size(); iQN++) {
              // Assemble the face terms
              const auto& xQN = qr_F.first[iQN];
              const auto& wQN = qr_F.second[iQN];
              auto g_xQN = boundary(iT, iF_loc, xQN, nTF);
              size_t offset_F = gfn[F] * element_type::nb_local_face_dofs;
              // For each face basis function
              for (size_t i = 0; i < element_type::nb_local_face_dofs; i++) {
                B(offset_F + i) += wQN * E.face_functions_on_face(iF_loc)(i, iQN) * g_xQN;
              }  // for i
            }    // for iQN
          }      // if
        }        // for it_F
      }

      // Assemble the linear system
      A.setFromTriplets(std::begin(triplets), end(triplets));
      A_sc.setFromTriplets(std::begin(triplets_sc), std::end(triplets_sc));
    
      // Solve the system
      iterative_solver_type solver;
      solver.analyzePattern(A);
      solver.factorize(A);
      if (solver.info() != Eigen::Success) {
        std::cerr << "Solver failed to factorise matrix. Returned code " << solver.info() << std::endl;
        exit(1);
      }
      dynamic_vector<scalar_type> xF = solver.solve(B);
      if (solver.info() != Eigen::Success) {
        std::cerr << "Solver failed to solve equation. Returned code " << solver.info() << std::endl;
        exit(1);
      }
      
      // Recover the cell unknowns
      dynamic_vector<scalar_type> XTFh = dynamic_vector<scalar_type>::Zero(nb_tot_dofs);
      XTFh.head(nb_cell_dofs) = B_sc - A_sc * xF.head(nb_face_dofs);
      XTFh.tail(nb_face_dofs) = xF;
      
      return XTFh;
    }
};

}  // namespace hho


