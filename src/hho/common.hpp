// file  : common.hpp
// author: Daniel Anderson
//
// Contains common HHO definitions.
//
#pragma once

#include <functional>

#include <boost/filesystem.hpp>
#include <Eigen/Dense>

#include "Mesh/eigen.hpp"
#include "Mesh/geometry.hpp"
#include "Mesh/geometry_generic.hpp"
#include "Mesh/loader.hpp"
#include "Mesh/mesh.hpp"

namespace hho {

// ----------------------------------------------------------------------------------------
//                              Generic mesh types
// ----------------------------------------------------------------------------------------

// A 2D mesh with double-valued coordinates
using mesh_type = generic_mesh<double, 2>;
using scalar_type = mesh_type::scalar_type;
using vector_type = static_vector<scalar_type, 2>;
using point_type = typename mesh_type::point_type;
using tensor_type = typename Eigen::Matrix<scalar_type, mesh_type::dimension, mesh_type::dimension>;

// ----------------------------------------------------------------------------------------
//                                Function types
// ----------------------------------------------------------------------------------------

// Standard functions of space
using scalar_function_of_space = typename std::function<scalar_type(point_type)>;
using vector_function_of_space = typename std::function<vector_type(point_type)>;
using tensor_function_of_space = typename std::function<tensor_type(point_type)>;

// Standard functions of time
using scalar_function_of_time = typename std::function<scalar_type(scalar_type)>;
using vector_function_of_time = typename std::function<vector_type(scalar_type)>;
using tensor_function_of_time = typename std::function<tensor_type(scalar_type)>;

// Cell-wise functions on the mesh -- takes a cell number and a point in the cell
using cellwise_scalar_function = typename std::function<scalar_type(size_t, point_type)>;
using cellwise_vector_function = typename std::function<vector_type(size_t, point_type)>;
using cellwise_tensor_function = typename std::function<tensor_type(size_t, point_type)>;

// Flux function -- takes a cell number, local face number, a point on the face and a normal vector
// Used for fluxes and Neumann boundary conditions
using cellwise_flux_function = typename std::function<scalar_type(size_t, size_t, point_type, vector_type)>;
using local_flux_function = typename std::function<scalar_type(size_t, point_type, vector_type)>;

// Functions of space and time
using scalar_function_of_spacetime = typename std::function<scalar_type(scalar_type, point_type)>;

// Functions of functions of space of time
using cellwise_scalar_function_of_time = typename std::function<cellwise_scalar_function(scalar_type)>;
using cellwise_vector_function_of_time = typename std::function<cellwise_vector_function(scalar_type)>;
using cellwise_tensor_function_of_time = typename std::function<cellwise_tensor_function(scalar_type)>;

using cellwise_flux_function_of_time = typename std::function<cellwise_flux_function(scalar_type)>;

// ----------------------------------------------------------------------------------------
//                                Utility functions
// ----------------------------------------------------------------------------------------

// Create a directory with the given path if it does not already
// exist, else do nothing
//
// Arguments:
//   path  the (boost) path to the required directory
void assert_path(const boost::filesystem::path& path) {
  if (!boost::filesystem::exists(path)) {
    assert_path(path.parent_path());  // ensure the parent exists first
    boost::filesystem::create_directory(path);
  }
  assert(boost::filesystem::exists(path));
}

// Create a directory with the given path name if it does
// not already exist, else do nothing
//
// Arguments:
//   name  the directory path to create if nonexistent
void assert_directory(const std::string& name) {
  assert_path(boost::filesystem::path(name));
}

}
