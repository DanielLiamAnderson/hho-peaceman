// file  : advection_reaction_operator.hpp
// author: Daniel Anderson
//
// The local advection-reaction operator a_{beta,mu}
//
#pragma once

#include "common.hpp"

#include "hybrid_element.hpp"
#include "operator.hpp"

namespace hho {

template <typename element_type>
class advection_reaction_operator
    : public basic_square_operator<element_type> {
 public:
 
 	using parent_type = basic_square_operator<element_type>;

  // Create a local advection-reaction operator
  //
  // Arguments:
  //  velocity    the local velocity
  //  flux        the local flux of the velocity as a function of local face numbers
  //  reaction    the local reaction terms
  //  nu          the local diffusion tensor
  advection_reaction_operator(const vector_function_of_space& velocity, const local_flux_function& flux,
    const scalar_function_of_space& reaction, const tensor_function_of_space& nu,
    const scalar_type& eta = 1., const size_t& offset_i = 0, const size_t& offset_j = 0)
      : parent_type(offset_i, offset_j), m_eta(eta), velocity(velocity), flux(flux), reaction(reaction), nu(nu) {
    // Do nothing
  }

  // Accessors
  const dynamic_matrix<scalar_type>& advective_derivative() const { return m_GBT; }

  // Compute local contribution
  std::pair<dynamic_matrix<scalar_type>, dynamic_vector<scalar_type>> compute(const element_type& E) {
    const auto& Th = E.mesh();
    const auto& T = E.cell();

    //------------------------------------------------------------------------------
    // Count local unknowns

    size_t nb_cell_dofs = element_type::nb_local_cell_dofs;
    size_t nb_face_dofs = element_type::nb_local_face_dofs;
    size_t nb_tot_dofs = E.nb_tot_dofs();

    //------------------------------------------------------------------------------
    // Initialize matrices

    // Discrete advective derivative G_beta,T
    dynamic_matrix<scalar_type> MG = dynamic_matrix<scalar_type>::Zero(nb_cell_dofs, nb_cell_dofs);
    dynamic_matrix<scalar_type> BG = dynamic_matrix<scalar_type>::Zero(nb_cell_dofs, nb_tot_dofs);

    // Stabilisation
    dynamic_matrix<scalar_type> SBT = dynamic_matrix<scalar_type>::Zero(nb_tot_dofs, nb_tot_dofs);

    // Reaction Mass matrix
    dynamic_matrix<scalar_type> MTT = dynamic_matrix<scalar_type>::Zero(nb_cell_dofs, nb_cell_dofs);

    // Operator matrix a_{beta, mu, h} and RHS b
    dynamic_matrix<scalar_type> ATF = dynamic_matrix<scalar_type>::Zero(nb_tot_dofs, nb_tot_dofs);
    dynamic_vector<scalar_type> bTF = dynamic_vector<scalar_type>::Zero(nb_tot_dofs);

    //------------------------------------------------------------------------------
    // Build and solve local problems

    auto xT = barycenter(Th, T);
    auto FT = faces(Th, T);
    
    //------------------------------------------------------------------------------
    // Assemble the local problems

    // For each face F in F_T
    for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
      auto F = *it_F;
      auto iF_loc = std::distance(FT.begin(), it_F);
      auto nTF = normal(Th, F, xT);
      auto hF = measure(Th, F);
      auto xF = barycenter(Th, F);

      const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);
      const auto& qr_F = E.quadrature_rule_on_face(iF_loc);

      //------------------------------------------------------------------------------
      // Volumetric terms

      // Cell-on-cell / Cell-on-cell terms
      for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
        const auto& xQN = qr_PTF.first[iQN];
        const auto& wQN = qr_PTF.second[iQN];
        const auto& beta_iqn = velocity(xQN);
        const auto& mu_iqn = reaction(xQN);

        for (size_t i = 0; i < nb_cell_dofs; i++) {
          const auto& phi_i_iqn = E.cell_functions_on_pyramid(iF_loc)(i, iQN);
          for (size_t j = 0; j < nb_cell_dofs; j++) {
            const auto& phi_j_iqn = E.cell_functions_on_pyramid(iF_loc)(j, iQN);
            const auto& dphi_j_iqn = E.cell_gradients_on_pyramid(iF_loc)(j, iQN);
            // Advective derivatie LHS (Mass matrix)
            MG(i, j) += wQN * phi_i_iqn * phi_j_iqn;
            // Advective derivative RHS (beta \cdot \nabla(vT), w)_T
            BG(i, j) += wQN * beta_iqn.dot(dphi_j_iqn) * phi_i_iqn;
            // Reaction matrix (mu * phi_i, phi_j)_T
            MTT(i, j) += wQN * phi_i_iqn * phi_j_iqn * mu_iqn;
          }  // for j
        }    // for i
      }      // for iQN

      //------------------------------------------------------------------------------
      // Interface terms

      // Offset for face unknowns
      size_t offset_F = element_type::nb_local_cell_dofs + iF_loc * element_type::nb_local_face_dofs;

      for (size_t iQN = 0; iQN < qr_F.first.size(); iQN++) {
        const auto& xQN = qr_F.first[iQN];
        const auto& wQN = qr_F.second[iQN];

        // beta \cdot n_TF (The flux)
        auto beta_nTF_iqn = flux(iF_loc, xQN, nTF);
        auto nu_F = nu(xF).template lpNorm<Eigen::Infinity>();

        // Upwinding
        auto Aminus_iqn = upwind_term(nu_F, hF, beta_nTF_iqn, -1);

        for (size_t i = 0; i < nb_cell_dofs; i++) {
          const auto& phi_i_iqn = E.cell_functions_on_face(iF_loc)(i, iQN);

          // Cell-on-face / Face-on-face terms
          for (size_t j = 0; j < nb_face_dofs; j++) {
            const auto& phi_j_iqn = E.face_functions_on_face(iF_loc)(j, iQN);

            // Advective derivative RHS ((beta \cdot n_TF)v_F, w)_F
            BG(i, offset_F + j) += wQN * beta_nTF_iqn * phi_i_iqn * phi_j_iqn;

            // Stability term -(A^- w_T, v_F)_F
            SBT(i, offset_F + j) -= wQN * Aminus_iqn * phi_i_iqn * phi_j_iqn;
            // Stability term -(A^- w_F, v_T)_F
            SBT(offset_F + j, i) -= wQN * Aminus_iqn * phi_i_iqn * phi_j_iqn;

          }  // for j

          // Cell-on-face / Cell-on-face terms
          for (size_t j = 0; j < nb_cell_dofs; j++) {
            const auto& phi_j_iqn = E.cell_functions_on_face(iF_loc)(j, iQN);

            // Advective derivative RHS -((beta \cdot n_TF)vT, w)_F
            BG(i, j) -= wQN * beta_nTF_iqn * phi_i_iqn * phi_j_iqn;

            // Stability term (A^- w_T, v_T)_F
            SBT(i, j) += wQN * Aminus_iqn * phi_i_iqn * phi_j_iqn;

          }  // for j
        }    // for i

        // Face-on-face / Face-on-face terms
        for (size_t i = 0; i < nb_face_dofs; i++) {
          const auto& phi_i_iqn = E.face_functions_on_face(iF_loc)(i, iQN);

          for (size_t j = 0; j < nb_face_dofs; j++) {
            const auto& phi_j_iqn = E.face_functions_on_face(iF_loc)(j, iQN);

            // Stability term (A^- w_F, v_F)_F
            SBT(offset_F + i, offset_F + j) += wQN * Aminus_iqn * phi_i_iqn * phi_j_iqn;
          }  // for j
        }    // for i

      }  // for iQN
    }    // for iF_loc

    // Solve for the advective-derivative operator
    m_GBT = MG.ldlt().solve(BG);
    
    // Assemble the local operator
    for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
      auto F = *it_F;
      auto iF_loc = std::distance(FT.begin(), it_F);

      const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);

      // Cell-on-cell / Cell-on-cell terms
      for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
        const auto& wQN = qr_PTF.second[iQN];

        for (size_t j = 0; j < nb_tot_dofs; j++) {
          // Compute G_beta_T w at xQN
          const auto G_beta_T_w = m_GBT.col(j);
          scalar_type G_beta_T_w_iqn = 0.0;
          for (size_t i = 0; i < nb_cell_dofs; i++) {
            const auto& phi_i_iqn = E.cell_functions_on_pyramid(iF_loc)(i, iQN);
            G_beta_T_w_iqn += G_beta_T_w(i) * phi_i_iqn;
          }
          
          for (size_t i = 0; i < nb_cell_dofs; i++) {
            const auto& phi_i_iqn = E.cell_functions_on_pyramid(iF_loc)(i, iQN);
            // Consistent term - (w_T, G_BT v)_T
            ATF(j, i) -= wQN * phi_i_iqn * G_beta_T_w_iqn;
          }
        }   
      }      
    }

    ATF.topLeftCorner(nb_cell_dofs, nb_cell_dofs) += MTT;  // (mu w_T, v_T)_T
    ATF += SBT;                                            // s^-(w, v)

    return std::make_pair(ATF, bTF);
  }

 private:
 
  // -------------------------------
  //     The upwinding schemes
 
  // The upwind scheme
  scalar_type upwind(const scalar_type s) {
    return std::abs(s);
  }

  // The locally upwind theta scheme
  scalar_type locally_upwind_theta(const scalar_type s) {
    if (std::abs(s) <= 0.5) return 0.0;
    else return std::abs(s);
  }

  // The Scharfetter Gummel scheme
  scalar_type scharfetter_gummel(const scalar_type s) {
    return s * (exp(s)+1.)/(exp(s)-1.) - 2.;
  }

  // Compute the upwinding term
  scalar_type upwind_term(const scalar_type nu_F, const scalar_type hF, const scalar_type beta_nTF, int sign) {
    // Degenerate case
    if (std::abs(nu_F) < 1e-12) {
      return 0.5 * (std::abs(beta_nTF) + sign * beta_nTF);
    } else {
      auto PeTF = hF * beta_nTF / nu_F;   // the local Peclet number
      auto Aabs = (nu_F / hF) * locally_upwind_theta(PeTF);
      auto A_PeTF = 0.5 * (Aabs + sign * beta_nTF);
      return A_PeTF;
    }
  }
 
  // Local problem data
  scalar_type m_eta;
  dynamic_matrix<scalar_type> m_GBT;
  vector_function_of_space velocity;
  local_flux_function flux;
  scalar_function_of_space reaction;
  tensor_function_of_space nu;
};

}  // namespace hho
