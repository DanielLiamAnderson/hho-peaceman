// file  : operator.hpp
// author: Daniele Di Pietro
#pragma once

#include "Mesh/eigen.hpp"

//------------------------------------------------------------------------------

template <typename ELEMENT1_TYPE, typename ELEMENT2_TYPE>
class basic_operator {
 public:
  typedef ELEMENT1_TYPE element1_type;
  typedef ELEMENT2_TYPE element2_type;

  typedef typename element1_type::scalar_type scalar_type;
  typedef typename element1_type::mesh_type mesh_type;

  basic_operator(size_t offset_i, size_t offset_j) : m_offset_i(offset_i), m_offset_j(offset_j) {
    static_assert(std::is_same<mesh_type, typename element2_type::mesh_type>::value,
                  "Incoherent mesh types in mixed operator");
  }

  inline const size_t& offset_i() const { return m_offset_i; }

  inline const size_t& offset_j() const { return m_offset_j; }

 protected:
  size_t m_offset_i;
  size_t m_offset_j;
};

//------------------------------------------------------------------------------

template <typename ELEMENT_TYPE>
class basic_square_operator : public basic_operator<ELEMENT_TYPE, ELEMENT_TYPE> {
 public:
  // Typedefs
  typedef basic_operator<ELEMENT_TYPE, ELEMENT_TYPE> parent_type;
  typedef typename parent_type::element1_type element_type;
  typedef typename parent_type::scalar_type scalar_type;
  typedef typename parent_type::mesh_type mesh_type;

  // Constructor
  basic_square_operator(size_t offset_i, size_t offset_j) : parent_type(offset_i, offset_j) {
    // Do nothing
  }

  // Virtual members
  virtual std::pair<dynamic_matrix<scalar_type>, dynamic_vector<scalar_type> > compute(const element_type& E) = 0;
};

//------------------------------------------------------------------------------

template <typename ELEMENT1_TYPE, typename ELEMENT2_TYPE>
class basic_rectangular_operator : public basic_operator<ELEMENT1_TYPE, ELEMENT2_TYPE> {
 public:
  // Typedefs
  typedef basic_operator<ELEMENT1_TYPE, ELEMENT2_TYPE> parent_type;
  typedef typename parent_type::element1_type element1_type;
  typedef typename parent_type::element2_type element2_type;
  typedef typename parent_type::scalar_type scalar_type;
  typedef typename parent_type::mesh_type mesh_type;

  // Constructor
  basic_rectangular_operator(size_t offset_i, size_t offset_j) : parent_type(offset_i, offset_j) {
    // Do nothing
  }

  // Virtual members
  virtual std::pair<dynamic_matrix<scalar_type>, dynamic_vector<scalar_type> > compute(const element1_type& E1,
                                                                                       const element2_type& E2) = 0;
};
