// file  : hybrid_element.hpp
// author: Daniele Di Pietro & Daniel Anderson
//
// The hybrid element class encapsulates the information required
// for numerical quadrature of the local polynomial basis functions
// of a given cell and its faces.
//
// The class precomputes and stores evalutations of:
//	- The cell basis functions over the quadrature points of each
// 	- triangular/pyramid subelement of the cell
//  - The cell basis functions of the quadrature points of the face
//  - The gradient basis functions over the quadrature points of each
// 	- triangular/pyramid subelement of the cell
//  - The gradient basis functions over the quadrature points of the face
//  - The face basis functions over the quadrature points of the face

#pragma once

#include "basic_element.hpp"

namespace hho {

template <typename _mesh_type, size_t L, size_t K>
class hybrid_element : public basic_element<_mesh_type> {
 public:
  // Types
  using mesh_type = _mesh_type;
  using parent_type = basic_element<mesh_type>;
  using scalar_type = typename mesh_type::scalar_type;
  using cell_type = typename mesh_type::cell;
  using face_type = typename mesh_type::face;
  using quadrature_rule = typename parent_type::quadrature_rule;

  static constexpr size_t cell_degree = L;
  static constexpr size_t face_degree = K;

  static constexpr size_t nb_local_cell_dofs = polynomial_basis<scalar_type, mesh_type::dimension, L>::type::size;
  static constexpr size_t nb_local_face_dofs = polynomial_basis<scalar_type, mesh_type::dimension - 1, K>::type::size;

	using cell_basis_type = typename polynomial_basis<scalar_type, mesh_type::dimension, cell_degree + 1>::type;
  using face_basis_type = typename polynomial_basis<scalar_type, mesh_type::dimension - 1, face_degree>::type;

  using cell_function_eval = BasisFunctionEvaluation<cell_basis_type>;
  using cell_gradient_eval = BasisGradientEvaluation<cell_basis_type>;
  using face_function_eval = BasisFunctionEvaluation<face_basis_type>;

  // Constructor
  hybrid_element(const mesh_type& Th, const cell_type& T, const int& cell_doe, const int& face_doe) : parent_type(Th, T, cell_doe, face_doe), m_cell_basis(barycenter(Th, T), diameter(Th, T)) {
		// Create bases
		m_face_basis.reserve(T.subelement_size());

		auto FT = faces(Th, T);
		for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
		  auto F = *it_F;

		  m_face_basis.push_back(
		      std::shared_ptr<face_basis_type>(new face_basis_type(points(Th, F)[0], barycenter(Th, F), measure(Th, F))));

		}  // for it_F

		// Evaluate basis at quadrature nodes
		m_cell_function_eval_pyramid.reserve(T.subelement_size());
		m_cell_gradient_eval_pyramid.reserve(T.subelement_size());
		m_cell_function_eval_face.reserve(T.subelement_size());
		m_cell_gradient_eval_face.reserve(T.subelement_size());
		m_face_function_eval_face.reserve(T.subelement_size());

		for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
		  size_t iF_loc = std::distance(FT.begin(), it_F);
		  assert(iF_loc < T.subelement_size());

		  auto F = *it_F;

		  const auto& qr_PTF = this->quadrature_rule_on_pyramid(iF_loc);
		  const auto& qr_F = this->quadrature_rule_on_face(iF_loc);

		  m_cell_function_eval_pyramid.push_back(
		      std::shared_ptr<cell_function_eval>(new cell_function_eval(&m_cell_basis, qr_PTF.first)));
		  m_cell_gradient_eval_pyramid.push_back(
		      std::shared_ptr<cell_gradient_eval>(new cell_gradient_eval(&m_cell_basis, qr_PTF.first)));

		  m_cell_function_eval_face.push_back(
		      std::shared_ptr<cell_function_eval>(new cell_function_eval(&m_cell_basis, qr_F.first)));
		  m_cell_gradient_eval_face.push_back(
		      std::shared_ptr<cell_gradient_eval>(new cell_gradient_eval(&m_cell_basis, qr_F.first)));

		  m_face_function_eval_face.push_back(
		      std::shared_ptr<face_function_eval>(new face_function_eval(m_face_basis[iF_loc].get(), qr_F.first)));
		}  // for it_F
	}

  // Accessors
  // Return the basis for the cell element
  const cell_basis_type& cell_basis() const { return m_cell_basis; }

  // Return the basis for the face element with given index
  //
  // Arguments:
  //  iF_loc		The index of the face
  //
  // Returns: The basis of the face
  const face_basis_type& face_basis(const size_t& iF_loc) const {
    assert(iF_loc < this->cell().subelement_size());
    return *m_face_basis[iF_loc];
  }

  // Return the number of degrees of freedom of a cell function
  inline const size_t nb_cell_dofs() const { return nb_local_cell_dofs; }

  // Return the number of degrees of freedom of the face functions
  // on this cell (that is the total number, not the number per face)
  inline const size_t nb_face_dofs() const { return this->cell().subelement_size() * nb_local_face_dofs; }

  // Return the total number of degrees of freedom of a set of local
  // functions (cell and face) on this cell.
  inline const size_t nb_tot_dofs() const { return nb_cell_dofs() + nb_face_dofs(); }

  // Return the basis function evaluation of the cell basis functions over the triangular
  // / pyramid subelement corresponding to the given face index
  //
  // Arguments:
  //  iF_loc		the index of the face corresponding to the cell subelement
  //
  // Returns: the cell basis function evaluation over the subelement
  inline const cell_function_eval& cell_functions_on_pyramid(const size_t& iF_loc) const {
    assert(iF_loc < this->cell().subelement_size());
    return *m_cell_function_eval_pyramid[iF_loc];
  }

  // Return the basis function evaluation of the gradient basis functions over the triangular
  // / pyramid subelement corresponding to the given face index
  //
  // Arguments:
  //  iF_loc		the index of the face corresponding to the cell subelement
  //
  // Returns: the gradient basis function evaluation over the subelement
  inline const cell_gradient_eval& cell_gradients_on_pyramid(const size_t& iF_loc) const {
    assert(iF_loc < this->cell().subelement_size());
    return *m_cell_gradient_eval_pyramid[iF_loc];
  }

  // Return the basis function evaluation of the cell basis functions over the face
  // with the given index
  //
  // Arguments:
  //  iF_loc		the index of the face
  //
  // Returns: the cell basis function evaluation over the face
  inline const cell_function_eval& cell_functions_on_face(const size_t& iF_loc) const {
    assert(iF_loc < this->cell().subelement_size());
    return *m_cell_function_eval_face[iF_loc];
  }

  // Returns the basis function evaluation of the gradient basis functions over the face
  // with the given index
  //
  // Arguments:
  //  iF_loc		the index of the face
  //
  // Returns: the gradient basis function evaluation over the face
  inline const cell_gradient_eval& cell_gradients_on_face(const size_t& iF_loc) const {
    assert(iF_loc < this->cell().subelement_size());
    return *m_cell_gradient_eval_face[iF_loc];
  }

  // Returns the basis function evaluation of the face basis function over the face
  // with the given index
  //
  // Arguments:
  //  iF_loc		the index of the face
  //
  // Returns: the face basis function evaluation over the face
  inline const face_function_eval& face_functions_on_face(const size_t& iF_loc) const {
    assert(iF_loc < this->cell().subelement_size());
    return *m_face_function_eval_face[iF_loc];
  }

 private:
  cell_basis_type m_cell_basis;
  std::vector<std::shared_ptr<face_basis_type> > m_face_basis;

  std::vector<quadrature_rule> m_qr_pyramid;
  std::vector<quadrature_rule> m_qr_face;

  std::vector<std::shared_ptr<cell_function_eval> > m_cell_function_eval_pyramid;
  std::vector<std::shared_ptr<cell_gradient_eval> > m_cell_gradient_eval_pyramid;

  std::vector<std::shared_ptr<cell_function_eval> > m_cell_function_eval_face;
  std::vector<std::shared_ptr<cell_gradient_eval> > m_cell_gradient_eval_face;

  std::vector<std::shared_ptr<face_function_eval> > m_face_function_eval_face;
};

}  // namespace hho
