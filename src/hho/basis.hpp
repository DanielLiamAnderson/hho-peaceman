// file  : basis.hpp
// author: Daniele Di Pietro
#pragma once

#include <array>
#include <functional>

#include <boost/math/special_functions/binomial.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/mpl/if.hpp>

#include <Eigen/Dense>

#include "Mesh/point.hpp"

namespace hho {

using boost::mpl::if_;
using boost::mpl::int_;
using boost::mpl::equal_to;

//------------------------------------------------------------------------------

namespace detail {
  template <unsigned int n>
  struct Factorial {
    enum { value = n * Factorial<n - 1>::value };
  };

  template <>
  struct Factorial<0> {
    enum { value = 1 };
  };

  template <unsigned int n, unsigned int k>
  struct Binomial {
    enum { value = Factorial<n>::value / (Factorial<n - k>::value * Factorial<k>::value) };
  };
}  // namespace detail

//------------------------------------------------------------------------------

template <typename T, int K>
class HierarchicalScalarBasis2d {
 public:
  typedef T scalar_type;

  typedef T ValueType;
  typedef Eigen::Matrix<T, 2, 1> GradientType;

  typedef std::function<ValueType(const point<T, 2> &)> BasisFunctionType;
  typedef std::function<GradientType(const point<T, 2> &)> BasisGradientType;

  struct BasisFunction {
    BasisFunctionType phi;
    BasisGradientType dphi;
  };

  enum { dim = 1, degree = K, size = detail::Binomial<K + 2, 2>::value };

  typedef std::array<BasisFunction, HierarchicalScalarBasis2d<T, K>::size> BasisFunctionArray;

  HierarchicalScalarBasis2d(const point<T, 2> &xT, const T &hT);

  inline const BasisFunctionArray &Phi() const { return m_Phi; }
  inline const BasisFunction &phi(int i) const { return m_Phi[i]; }
  std::size_t degreeIndex(const std::size_t &k) const { return m_degree_index(k); }

 private:
  point<T, 2> m_xT;
  T m_hT;
  BasisFunctionArray m_Phi;
  Eigen::Array<std::size_t, HierarchicalScalarBasis2d<T, K>::size + 1, 1> m_degree_index;
};

//------------------------------------------------------------------------------

template <typename T, int K>
class HierarchicalScalarBasis1d {
 public:
  typedef T scalar_type;

  typedef T ValueType;
  typedef std::function<ValueType(const point<T, 2> &)> BasisFunctionType;

  struct BasisFunction {
    BasisFunctionType phi;
  };

  enum { degree = K, size = K + 1 };

  typedef std::array<BasisFunction, HierarchicalScalarBasis1d<T, K>::size> BasisFunctionArray;

  HierarchicalScalarBasis1d(const point<T, 2> &x0, const point<T, 2> &xF, const T &hF);

  inline const BasisFunctionArray &Phi() const { return m_Phi; }
  inline const BasisFunction &phi(std::size_t i) const { return m_Phi[i]; };
  std::size_t degreeIndex(const std::size_t &k) const { return k; }

 private:
  point<T, 2> m_x0;
  point<T, 2> m_xF;
  T m_hF;
  BasisFunctionArray m_Phi;
};

//------------------------------------------------------------------------------

// Warning: first and last here refer to the polynomial degree instead of the
// index of the basis function

template <typename BasisType>
class BasisFunctionEvaluation {
 public:
  typedef Eigen::Matrix<typename BasisType::ValueType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> BasisStorageType;

  BasisFunctionEvaluation(const BasisType *B, const std::vector<point<typename BasisType::scalar_type, 2>> &points,
                          std::size_t first = 0, std::size_t last = BasisType::degree);

  inline const typename BasisType::ValueType &operator()(std::size_t i, std::size_t j) const { return m_phi(i, j); }

  inline const typename BasisType::ValueType &phi(std::size_t i, std::size_t j) const { return m_phi(i, j); }

  inline const BasisStorageType &operator()() const { return m_phi; }

  inline const BasisStorageType &phi() const { return m_phi; }

 private:
  BasisStorageType m_phi;
};

//------------------------------------------------------------------------------

template <typename BasisType>
class BasisGradientEvaluation {
 public:
  BasisGradientEvaluation(const BasisType *B, const std::vector<point<typename BasisType::scalar_type, 2>> &points,
                          std::size_t first = 0, std::size_t last = BasisType::degree);

  inline const typename BasisType::GradientType &operator()(std::size_t i, std::size_t j) const { return m_dphi(i, j); }

 private:
  Eigen::Array<typename BasisType::GradientType, Eigen::Dynamic, Eigen::Dynamic> m_dphi;
};

//------------------------------------------------------------------------------
// Easy access

template <typename T, size_t D, int K>
struct polynomial_basis {
  typedef typename if_<equal_to<int_<D>, int_<2>>,
          HierarchicalScalarBasis2d<T, K>,
          HierarchicalScalarBasis1d<T, K>>::type type;
};


//------------------------------------------------------------------------------
// Implementation

//------------------------------------------------------------------------------
// HierarchicalScalarBasis2d

template <typename T, int K>
HierarchicalScalarBasis2d<T, K>::HierarchicalScalarBasis2d(const point<T, 2> &xT, const T &hT) : m_xT(xT), m_hT(hT) {
  int i_phi = 0;
  for (int k = 0; k <= K; k++) {
    m_degree_index(k) = i_phi;
    for (int i = 0; i <= k; i++, i_phi++) {
      BasisFunctionType phi = [i, k, xT, hT](const point<T, 2> &x) -> T {
        T _x = (x[0] - xT[0]) / hT;
        T _y = (x[1] - xT[1]) / hT;
        return std::pow(_x, i) * std::pow(_y, k - i);
      };
      BasisGradientType dphi = [i, k, xT, hT](const point<T, 2> &x) -> GradientType {
        GradientType g;
        T _x = (x[0] - xT[0]) / hT;
        T _y = (x[1] - xT[1]) / hT;
        g(0) = (i == 0) ? 0. : i * std::pow(_x, i - 1) / hT * std::pow(_y, k - i);
        g(1) = (i == k) ? 0. : std::pow(_x, i) * (k - i) * std::pow(_y, k - i - 1) / hT;
        return g;
      };
      m_Phi[i_phi].phi = phi;
      m_Phi[i_phi].dphi = dphi;
    }  // for i
  }    // for k
  m_degree_index(K + 1) = i_phi;
}

//------------------------------------------------------------------------------
// HierarchicalScalarBasis1d

template <typename T, int K>
HierarchicalScalarBasis1d<T, K>::HierarchicalScalarBasis1d(const point<T, 2> &x0, const point<T, 2> &xF, const T &hF)
    : m_x0(x0), m_xF(xF), m_hF(hF) {
  // Degree
  for (std::size_t k = 0; k <= K; k++) {
    BasisFunctionType phi = [k, this](const point<T, 2> &x) -> ValueType {
      T d = (x.to_vector() - this->m_xF.to_vector()).dot(this->m_x0.to_vector() - this->m_xF.to_vector()) /
            (std::pow(this->m_hF, 2));
      return std::pow(d, k);
    };
    m_Phi[k].phi = phi;
  }  // for k
}

//------------------------------------------------------------------------------
// BasisFunctionEvaluation

template <typename BasisType>
BasisFunctionEvaluation<BasisType>::BasisFunctionEvaluation(
    const BasisType *B, const std::vector<point<typename BasisType::scalar_type, 2>> &points, std::size_t first,
    std::size_t last) {
  assert(last >= first && first >= 0 && last <= BasisType::degree);

  int offset = B->degreeIndex(last + 1) - B->degreeIndex(first);
  int nb_points = points.size();

  m_phi.resize(offset, nb_points);

  for (int l = 0; l < nb_points; l++) {
    const point<typename BasisType::scalar_type, 2> &xl = points[l];

    std::size_t i_phi = 0;
    for (std::size_t i = B->degreeIndex(first); i < B->degreeIndex(last + 1); i++, i_phi++) {
      m_phi(i_phi, l) = B->phi(i).phi(xl);
    }  // for i
  }    // for l
}

//------------------------------------------------------------------------------
// BasisFunctionEvaluation

template <typename BasisType>
BasisGradientEvaluation<BasisType>::BasisGradientEvaluation(
    const BasisType *B, const std::vector<point<typename BasisType::scalar_type, 2>> &points, std::size_t first,
    std::size_t last) {
  assert(last >= first && first >= 0 && last <= BasisType::degree);

  int offset = B->degreeIndex(last + 1) - B->degreeIndex(first);
  int nb_points = points.size();

  m_dphi.resize(offset, nb_points);

  for (int l = 0; l < nb_points; l++) {
    const point<typename BasisType::scalar_type, 2> &xl = points[l];

    std::size_t i_phi = 0;
    for (std::size_t i = B->degreeIndex(first); i < B->degreeIndex(last + 1); i++, i_phi++) {
      m_dphi(i_phi, l) = B->phi(i).dphi(xl);
    }  // for i
  }    // for l
}

}  // namespace hho
