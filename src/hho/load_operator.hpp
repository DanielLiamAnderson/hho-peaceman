// file  : load_operator.hpp
// author: Daniele Di Pietro & Daniel Anderson
#pragma once

#include "hybrid_element.hpp"

namespace hho {

template <typename element_type>
class load_operator
    : public basic_square_operator<element_type> {
 public:
  
  using parent_type = basic_square_operator<element_type>;
  using mesh_type = typename parent_type::mesh_type;
  using scalar_type = typename mesh_type::scalar_type;
 
  using load_type = std::function<scalar_type(const typename mesh_type::point_type &)>;

  load_operator(const load_type &f, const size_t &offset_i = 0, const size_t &offset_j = 0)
      : parent_type(offset_i, offset_j), m_f(f) {
    // Do nothing
  }

  std::pair<dynamic_matrix<scalar_type>, dynamic_vector<scalar_type> > compute(const element_type &E) {
    const auto &Th = E.mesh();
		const auto &T = E.cell();

		dynamic_matrix<scalar_type> ATF;
		dynamic_vector<scalar_type> bTF = dynamic_vector<scalar_type>::Zero(element_type::nb_local_cell_dofs);

		auto FT = faces(Th, T);
		for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
		  auto F = *it_F;
		  auto iF_loc = std::distance(FT.begin(), it_F);

		  const auto &qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);

		  for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
		    const auto &xQN = qr_PTF.first[iQN];
		    const auto &wQN = qr_PTF.second[iQN];

		    auto f_xQN = m_f(xQN);

		    for (size_t i = 0; i < element_type::nb_local_cell_dofs; i++) {
		      bTF(i) += wQN * E.cell_functions_on_pyramid(iF_loc)(i, iQN) * f_xQN;
		    }  // for i
		  }    // for iQN
		}

		return std::make_pair(ATF, bTF);
  }

 private:
  load_type m_f;
};

}  // namespace hho
