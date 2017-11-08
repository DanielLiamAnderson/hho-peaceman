// file  : interpolation_operator.hpp
// author: Daniele Di Pietro
#pragma once

#include <boost/mpl/comparison.hpp>
#include <boost/mpl/if.hpp>

#include "hybrid_element.hpp"
#include "operator.hpp"

namespace hho {

template <typename element_type>
class interpolation_operator
    : public basic_square_operator<element_type> {
 public:
 
  using parent_type = basic_square_operator<element_type>;
  using mesh_type = typename parent_type::mesh_type;
  using scalar_type = typename mesh_type::scalar_type; 
 
  using interpoland_type = std::function<scalar_type(const typename mesh_type::point_type &)>;
 
  interpolation_operator(const interpoland_type &v, const size_t &offset_i = 0, const size_t &offset_j = 0)
      : parent_type(offset_i, offset_j), m_v(v) {
    // Do nothing
  }

  std::pair<dynamic_matrix<scalar_type>, dynamic_vector<scalar_type> > compute(const element_type &E) {
  	const auto &Th = E.mesh();
		const auto &T = E.cell();

		// Compute mass matrix
		dynamic_matrix<scalar_type> ATF = dynamic_matrix<scalar_type>::Zero(E.nb_tot_dofs(), E.nb_tot_dofs());
		dynamic_vector<scalar_type> bTF = dynamic_vector<scalar_type>::Zero(E.nb_tot_dofs());

		auto FT = faces(Th, T);
		for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
		  auto F = *it_F;
		  auto iF_loc = std::distance(FT.begin(), it_F);

		  const auto &qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);
		  const auto &qr_F = E.quadrature_rule_on_face(iF_loc);

		  // Volumetric terms
		  for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
		    const auto &xQN = qr_PTF.first[iQN];
		    const auto &wQN = qr_PTF.second[iQN];

		    for (std::size_t i = 0; i < element_type::nb_local_cell_dofs; i++) {
		      const auto &phi_i_iqn = E.cell_functions_on_pyramid(iF_loc)(i, iQN);
		      for (std::size_t j = 0; j < element_type::nb_local_cell_dofs; j++) {
		        const auto &phi_j_iqn = E.cell_functions_on_pyramid(iF_loc)(j, iQN);
		        ATF(i, j) += wQN * phi_i_iqn * phi_j_iqn;
		      }  // for j
		      bTF(i) += wQN * phi_i_iqn * m_v(xQN);
		    }  // for i

		  }  // for iQN

		  // Boundary terms
		  for (size_t iQN = 0; iQN < qr_F.first.size(); iQN++) {
		    const auto &xQN = qr_F.first[iQN];
		    const auto &wQN = qr_F.second[iQN];

		    // Offset for face unknowns
		    size_t offset_F = element_type::nb_local_cell_dofs + iF_loc * element_type::nb_local_face_dofs;

		    for (size_t i = 0; i < element_type::nb_local_face_dofs; i++) {
		      const auto &phi_i_iqn = E.face_functions_on_face(iF_loc)(i, iQN);
		      for (size_t j = 0; j < element_type::nb_local_face_dofs; j++) {
		        const auto &phi_j_iqn = E.face_functions_on_face(iF_loc)(j, iQN);
		        ATF(offset_F + i, offset_F + j) += wQN * phi_i_iqn * phi_j_iqn;
		      }  // for j
		      bTF(offset_F + i) += wQN * phi_i_iqn * m_v(xQN);
		    }  // for i

		  }  // for iQN

		}  // for iF_loc

		return std::make_pair(ATF, bTF);
		
  }

 private:
  interpoland_type m_v;
};

}  // namespace hho
