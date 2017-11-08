// file  : basic_element.hpp
// author: Daniele Di Pietro
//
// The basic element class encapsulates the information required to perform
// numerical quadrature on generic functions defined on a cell and its adjacent
// faces.

#pragma once

#include "basis.hpp"
#include "Mesh/geometry_generic.hpp"

namespace hho {

template <typename mesh_type>
class basic_element {
 public:
  // Types
  using scalar_type = typename mesh_type::scalar_type;
  using cell_type = typename mesh_type::cell;
  using face_type = typename mesh_type::face;

  // A quadrature rule is a weighted set of points, representing the numerical
  // weighting given to the specified points for numerical quadrature.
  using quadrature_rule = std::pair<std::vector<point<scalar_type, mesh_type::dimension>>, std::vector<scalar_type>>;

  // Constructor
  basic_element(const mesh_type& Th, const cell_type& T, const int& cell_doe, const int& face_doe)
      : m_Th(Th), m_T(T), m_cell_doe(cell_doe), m_face_doe(face_doe) {
    // Generate quadrature rules
    m_qr_pyramid.reserve(m_T.subelement_size());
    m_qr_face.reserve(m_T.subelement_size());

    auto FT = faces(m_Th, m_T);
    for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
      auto F = *it_F;

      auto qr_PTF = integrate(m_Th, m_T, F, m_cell_doe);
      auto qr_F = integrate(m_Th, F, m_face_doe);

      m_qr_pyramid.push_back(qr_PTF);
      m_qr_face.push_back(qr_F);
    }  // for it_F
  }

  // Accessors
  // Return a reference to the mesh that this element belongs to
  const mesh_type& mesh() const { return m_Th; }

  // Return a reference to the cell that this element is defined on
  const cell_type& cell() const { return m_T; }

  // Return a reference to the quadrature rule for the given local subelement
  //
  // Arguments:
  //  iF_loc		The index of the subelement
  //
  // Returns: a reference to the qudrature rule
  const quadrature_rule& quadrature_rule_on_pyramid(const size_t& iF_loc) const {
    assert(iF_loc < this->m_T.subelement_size());
    return m_qr_pyramid[iF_loc];
  }

  // Return a reference to the quadrature rule for the given local face
  //
  // Arguments:
  //  if_loc		The local index of the face
  //
  // Returns: a reference to the quadrature rule
  const quadrature_rule& quadrature_rule_on_face(const size_t& iF_loc) const {
    assert(iF_loc < this->m_T.subelement_size());
    return m_qr_face[iF_loc];
  }

 protected:
  const mesh_type& m_Th;
  const cell_type& m_T;

  int m_cell_doe;
  int m_face_doe;

  std::vector<quadrature_rule> m_qr_pyramid;
  std::vector<quadrature_rule> m_qr_face;
};

}  // namespace hho
