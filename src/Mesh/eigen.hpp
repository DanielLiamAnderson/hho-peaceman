#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

template<typename T>
using dynamic_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
using dynamic_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template<typename T, size_t M, size_t N>
using static_matrix = Eigen::Matrix<T, M, N>;

template<typename T, size_t N>
using static_vector = Eigen::Matrix<T, N, 1>;

template<typename T>
using sparse_matrix = Eigen::SparseMatrix<T>;

template<typename T>
using triplet = Eigen::Triplet<T>;

template<typename T>
static_vector<T, 3>
cross(const static_vector<T, 2>& v1, const static_vector<T, 2>& v2)
{
    static_vector<T, 3> ret;

    ret(0) = T(0);
    ret(1) = T(0);
    ret(2) = v1(0)*v2(1) - v1(1)*v2(0);

    return ret;
}
