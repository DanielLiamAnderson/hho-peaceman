// file  : variable_diffusion_operator.hpp
// author: Daniele Di Pietro & Daniel Anderson
#pragma once

#include "common.hpp"

#include "hybrid_element.hpp"
#include "operator.hpp"

namespace hho {

template <typename element_type>
class variable_diffusion_operator
    : public basic_square_operator<element_type> {
 public:
 
  using parent_type = basic_square_operator<element_type>;

  static constexpr size_t dim_grad_space = element_type::cell_basis_type::size - 1;

  variable_diffusion_operator(const tensor_function_of_space& diffusion_tensor, const scalar_type& eta = 1., const size_t& offset_i = 0, const size_t& offset_j = 0)
      : parent_type(offset_i, offset_j), m_eta(eta), m_diffusion_tensor(diffusion_tensor) {
    // Do nothing
  }

  // Accessors
  const dynamic_matrix<scalar_type>& gradient_reconstruction() const { return m_GT; }
  const dynamic_matrix<scalar_type>& stabilization() const { return m_STF; }

  // Compute local contribution
  std::pair<dynamic_matrix<scalar_type>, dynamic_vector<scalar_type>> compute(const element_type& E) {
    const auto& Th = E.mesh();
    const auto& T = E.cell();

    //------------------------------------------------------------------------------
    // Count local unknowns

    size_t nb_tot_dofs = E.nb_tot_dofs();

    //------------------------------------------------------------------------------
    // Initialize matrices

    dynamic_matrix<scalar_type> MG = dynamic_matrix<scalar_type>::Zero(dim_grad_space, dim_grad_space);
    dynamic_matrix<scalar_type> BG = dynamic_matrix<scalar_type>::Zero(dim_grad_space, nb_tot_dofs);

    Eigen::Matrix<scalar_type, element_type::cell_basis_type::size, element_type::cell_basis_type::size> MTT =
        Eigen::Matrix<scalar_type, element_type::cell_basis_type::size, element_type::cell_basis_type::size>::Zero();

    std::vector<dynamic_matrix<scalar_type> > MFF(T.subelement_size());
    std::vector<Eigen::Matrix<scalar_type, element_type::nb_local_face_dofs, element_type::cell_basis_type::size> > MFT(
        T.subelement_size());

    //------------------------------------------------------------------------------
    // Build and solve local problems

    auto xT = barycenter(Th, T);
    auto FT = faces(Th, T);

    for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
      auto F = *it_F;
      auto iF_loc = std::distance(FT.begin(), it_F);
      auto nTF = normal(Th, F, xT);

      const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);
      const auto& qr_F = E.quadrature_rule_on_face(iF_loc);

      //------------------------------------------------------------------------------
      // Initialize matrices

      MFT[iF_loc] = Eigen::Matrix<scalar_type, element_type::nb_local_face_dofs, element_type::cell_basis_type::size>::Zero();
      MFF[iF_loc] = Eigen::Matrix<scalar_type, element_type::nb_local_face_dofs, element_type::nb_local_face_dofs>::Zero();

      //------------------------------------------------------------------------------
      // Volumetric terms

      for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
        const auto& xQN = qr_PTF.first[iQN];
        const auto& wQN = qr_PTF.second[iQN];
        const auto kappa_iqn = m_diffusion_tensor(xQN);  // Evaluate the diffusion tensor

        for (size_t i = 0; i < dim_grad_space; i++) {
          const auto& dphi_i_iqn = E.cell_gradients_on_pyramid(iF_loc)(E.cell_basis().degreeIndex(1) + i, iQN);

          // LHS
          for (std::size_t j = 0; j < dim_grad_space; j++) {
            const auto& dphi_j_iqn = E.cell_gradients_on_pyramid(iF_loc)(E.cell_basis().degreeIndex(1) + j, iQN);

            MG(i, j) += wQN * (kappa_iqn * dphi_i_iqn).dot(dphi_j_iqn);
          }  // for j

          // RHS (\GRAD vT, \GRAD w)_{PTF}
          for (std::size_t j = 0; j < element_type::nb_local_cell_dofs; j++) {
            const auto& dphi_j_iqn = E.cell_gradients_on_pyramid(iF_loc)(j, iQN);

            BG(i, j) += wQN * (kappa_iqn * dphi_i_iqn).dot(dphi_j_iqn);
          }  // for j
        }    // for i

        // MTT
        for (std::size_t i = 0; i < element_type::cell_basis_type::size; i++) {
          const auto& phi_i_iqn = E.cell_functions_on_pyramid(iF_loc)(i, iQN);

          for (std::size_t j = 0; j < element_type::cell_basis_type::size; j++) {
            const auto& phi_j_iqn = E.cell_functions_on_pyramid(iF_loc)(j, iQN);

            MTT(i, j) += wQN * phi_i_iqn * phi_j_iqn;
          }  // for j
        }    // for i

      }  // for iQN

      //------------------------------------------------------------------------------
      // Boundary terms

      // Offset for face unknowns
      size_t offset_F = element_type::nb_local_cell_dofs + iF_loc * element_type::nb_local_face_dofs;

      for (size_t iQN = 0; iQN < qr_F.first.size(); iQN++) {
        const auto& xQN = qr_F.first[iQN];
        const auto& wQN = qr_F.second[iQN];

        auto kappa_iqn = m_diffusion_tensor(xQN);

        for (size_t i = 0; i < dim_grad_space; i++) {
          const auto& dphi_i_iqn = E.cell_gradients_on_face(iF_loc)(E.cell_basis().degreeIndex(1) + i, iQN);
          auto dphi_i_n_iqn = dphi_i_iqn.dot(kappa_iqn * nTF);

          // RHS (v_F, \GRAD w \dot K n_{TF})_F
          for (size_t j = 0; j < element_type::nb_local_face_dofs; j++) {
            const auto& phi_j_iqn = E.face_functions_on_face(iF_loc)(j, iQN);

            BG(i, offset_F + j) += wQN * dphi_i_n_iqn * phi_j_iqn;
          }  // for j

          // RHS -(v_T, \GRAD w \dot K n_{TF})_F
          for (size_t j = 0; j < element_type::nb_local_cell_dofs; j++) {
            const auto& phi_j_iqn = E.cell_functions_on_face(iF_loc)(j, iQN);

            BG(i, j) -= wQN * dphi_i_n_iqn * phi_j_iqn;
          }  // for j
        }    // for i

        // MFT
        for (size_t i = 0; i < element_type::nb_local_face_dofs; i++) {
          const auto& phi_i_iqn = E.face_functions_on_face(iF_loc)(i, iQN);
          for (size_t j = 0; j < element_type::cell_basis_type::size; j++) {
            const auto& phi_j_iqn = E.cell_functions_on_face(iF_loc)(j, iQN);

            MFT[iF_loc](i, j) += wQN * phi_i_iqn * phi_j_iqn;
          }  // for j
        }    // for i

        // MFF
        for (size_t i = 0; i < element_type::nb_local_face_dofs; i++) {
          const auto& phi_i_iqn = E.face_functions_on_face(iF_loc)(i, iQN);
          for (size_t j = 0; j < element_type::nb_local_face_dofs; j++) {
            const auto& phi_j_iqn = E.face_functions_on_face(iF_loc)(j, iQN);

            MFF[iF_loc](i, j) += wQN * phi_i_iqn * phi_j_iqn;
          }  // for j
        }    // for i
      }      // for iQN

    }  // for iF_loc

    //------------------------------------------------------------------------------
    // Consistent terms

    m_GT = MG.ldlt().solve(BG);
    dynamic_matrix<scalar_type> ATF = BG.transpose() * m_GT;

    //------------------------------------------------------------------------------
    // Stabilization

    m_STF = dynamic_matrix<scalar_type>::Zero(nb_tot_dofs, nb_tot_dofs);
    dynamic_matrix<scalar_type> piTL_rTK;
    {
      Eigen::LDLT<dynamic_matrix<scalar_type> > piLT;

      piLT.compute(MTT.topLeftCorner(E.cell_basis().degreeIndex(element_type::cell_degree + 1),
                                     E.cell_basis().degreeIndex(element_type::cell_degree + 1)));

      piTL_rTK = piLT.solve(MTT.block(0, E.cell_basis().degreeIndex(1),
                                      E.cell_basis().degreeIndex(element_type::cell_degree + 1), dim_grad_space) *
                            m_GT);
    }

    // Compute face residual
    for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
      auto F = *it_F;
      auto iF_loc = std::distance(FT.begin(), it_F);
      auto hF = measure(Th, F);

      auto xF = barycenter(Th, F);
      auto kappa_F = m_diffusion_tensor(xF).trace();

      dynamic_matrix<scalar_type> piF_rTK_minus_uF;
      dynamic_matrix<scalar_type> piF_uT_minus_piTL_rTK;
      {
        Eigen::LDLT<dynamic_matrix<scalar_type> > piKF;
        piKF.compute(MFF[iF_loc]);

        piF_rTK_minus_uF = piKF.solve(
            MFT[iF_loc].block(0, E.cell_basis().degreeIndex(1), element_type::nb_local_face_dofs, dim_grad_space) * m_GT);
        piF_rTK_minus_uF.block(0, element_type::nb_local_cell_dofs + iF_loc * element_type::nb_local_face_dofs,
                               element_type::nb_local_face_dofs, element_type::nb_local_face_dofs) -=
            dynamic_matrix<scalar_type>::Identity(element_type::nb_local_face_dofs, element_type::nb_local_face_dofs);

        dynamic_matrix<scalar_type> uT_minus_piTL_rTK = -piTL_rTK;
        uT_minus_piTL_rTK.topLeftCorner(element_type::nb_local_cell_dofs, element_type::nb_local_cell_dofs) +=
            dynamic_matrix<scalar_type>::Identity(element_type::nb_local_cell_dofs, element_type::nb_local_cell_dofs);
        piF_uT_minus_piTL_rTK =
            piKF.solve(MFT[iF_loc].topLeftCorner(element_type::nb_local_face_dofs,
                                                 E.cell_basis().degreeIndex(element_type::cell_degree + 1)) *
                       uT_minus_piTL_rTK);
      }

      dynamic_matrix<scalar_type> BRF = piF_uT_minus_piTL_rTK + piF_rTK_minus_uF;

      m_STF += kappa_F * (m_eta / hF) * BRF.transpose() * MFF[iF_loc] * BRF;
    }  // for iF_loc

    ATF += m_STF;

    dynamic_vector<scalar_type> bTF = dynamic_vector<scalar_type>::Zero(nb_tot_dofs);

    return std::make_pair(ATF, bTF);
  }

 private:
  scalar_type m_eta;
  dynamic_matrix<scalar_type> m_GT;
  dynamic_matrix<scalar_type> m_STF;
  tensor_function_of_space m_diffusion_tensor;
};

}  // namespace hho
