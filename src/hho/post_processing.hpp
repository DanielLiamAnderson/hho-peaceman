// file  : post_processing.hpp
// author: Daniele Di Pietro & Daniel Anderson
#pragma once

#include <boost/filesystem.hpp>

#include <Eigen/Dense>

#include "Mesh/mesh.hpp"

#include "basis.hpp"
#include "common.hpp"
#include "hybrid_element.hpp"

namespace hho {

namespace priv {

  template <typename T, std::size_t N>
  struct plot_nodes {
    typedef Eigen::Matrix<T, 2, 1> node_type;
    typedef Eigen::Matrix<std::size_t, 3, 1> connectivity_type;
  };

  template <typename T>
  struct plot_nodes<T, 0> {
    static const std::size_t nb_nodes = 3;
    static const std::size_t nb_subel = 1;

    typedef Eigen::Matrix<T, 2, 1> node_type;
    typedef Eigen::Matrix<std::size_t, 3, 1> connectivity_type;

    std::array<node_type, nb_nodes> nodes;
    std::array<connectivity_type, nb_subel> connectivity;

    plot_nodes() {
      nodes[0] << 0., 0.;
      nodes[1] << 1., 0.;
      nodes[2] << 0., 1.;

      connectivity[0] << 0, 1, 2;
    }
  };

  template <typename T>
  struct plot_nodes<T, 1> {
    static const std::size_t nb_nodes = 3;
    static const std::size_t nb_subel = 1;

    typedef Eigen::Matrix<T, 2, 1> node_type;
    typedef Eigen::Matrix<std::size_t, 3, 1> connectivity_type;

    std::array<node_type, nb_nodes> nodes;
    std::array<connectivity_type, nb_subel> connectivity;

    plot_nodes() {
      nodes[0] << 0., 0.;
      nodes[1] << 1., 0.;
      nodes[2] << 0., 1.;

      connectivity[0] << 0, 1, 2;
    }
  };

  template <typename T>
  struct plot_nodes<T, 2> {
    static const std::size_t nb_nodes = 6;
    static const std::size_t nb_subel = 4;

    typedef Eigen::Matrix<T, 2, 1> node_type;
    typedef Eigen::Matrix<std::size_t, 3, 1> connectivity_type;

    std::array<node_type, nb_nodes> nodes;
    std::array<connectivity_type, nb_subel> connectivity;

    plot_nodes() {
      nodes[0] << 0., 0.;
      nodes[1] << 1., 0.;
      nodes[2] << 0., 1.;
      nodes[3] << 0.5, 0.;
      nodes[4] << 0.5, 0.5;
      nodes[5] << 0, 0.5;

      connectivity[0] << 0, 3, 5;
      connectivity[1] << 3, 1, 4;
      connectivity[2] << 3, 4, 5;
      connectivity[3] << 5, 4, 2;
    }
  };

  template <typename T>
  struct plot_nodes<T, 3> {
    static const std::size_t nb_nodes = 10;
    static const std::size_t nb_subel = 9;

    typedef Eigen::Matrix<T, 2, 1> node_type;
    typedef Eigen::Matrix<std::size_t, 3, 1> connectivity_type;

    std::array<node_type, nb_nodes> nodes;
    std::array<connectivity_type, nb_subel> connectivity;

    plot_nodes() {
      nodes[0] << 0., 0.;
      nodes[1] << 1., 0.;
      nodes[2] << 0., 1.;
      nodes[3] << 1. / 3., 0.;
      nodes[4] << 2. / 3., 0.;
      nodes[5] << 2. / 3., 1. / 3.;
      nodes[6] << 1. / 3., 2. / 3.;
      nodes[7] << 0., 2. / 3.;
      nodes[8] << 0., 1. / 3.;
      nodes[9] << 1. / 3., 1. / 3.;

      connectivity[0] << 0, 3, 8;
      connectivity[1] << 3, 4, 9;
      connectivity[2] << 3, 9, 8;
      connectivity[3] << 8, 9, 7;
      connectivity[4] << 4, 1, 5;
      connectivity[5] << 4, 5, 9;
      connectivity[6] << 9, 5, 6;
      connectivity[7] << 9, 6, 7;
      connectivity[8] << 7, 6, 2;
    }
  };
}  // namespace priv

//------------------------------------------------------------------------------

template <size_t K, typename MESH_TYPE>
void post_processing(const char* file_name, const MESH_TYPE& Th,
                     const dynamic_vector<typename MESH_TYPE::scalar_type>& coeffs, const std::string& variable_name = "U") {
  typedef MESH_TYPE mesh_type;
  typedef typename mesh_type::scalar_type scalar_type;
  typedef hybrid_element<MESH_TYPE, K, K> element_type;
  typedef priv::plot_nodes<scalar_type, std::min<unsigned>(K+1,3)> plot_nodes_type;

  plot_nodes_type plot_nodes;

  std::array<std::stringstream, element_type::cell_basis_type::dim> ss_variable;
  for (std::size_t i = 0; i < element_type::cell_basis_type::dim; i++) {
    ss_variable[i] << "<DataArray type=\"Float32\" Name=\"" << variable_name << i << "\" format=\"ascii\">" << std::endl;
  }  // for i

  std::stringstream ss_points;
  ss_points << "<Points>" << std::endl;
  ss_points << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;

  std::stringstream ss_connectivity;
  ss_connectivity << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;

  std::size_t nb_points = 0, nb_cells = 0;

  for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
    const auto& T = *it_T;
    size_t iT = std::distance(Th.cells_begin(), it_T);
    auto xT = barycenter(Th, T);

    element_type E(Th, T, 0, 0);

    auto& coeffs_T = coeffs.segment(iT * element_type::nb_local_cell_dofs, element_type::nb_local_cell_dofs);

    auto FT = faces(Th, T);
    for (const auto& F : FT) {
      auto points_F = points(Th, F);

      auto geo_map = [ xT, points_F ](const typename plot_nodes_type::node_type& x) -> typename mesh_type::point_type {
        const auto& _xT = xT.to_vector();
        auto hat_x = _xT + (points_F[0].to_vector() - _xT) * x(0) + (points_F[1].to_vector() - _xT) * x(1);
        typename mesh_type::point_type res({hat_x(0), hat_x(1)});
        return res;
      };

      for (std::size_t iC = 0; iC < plot_nodes_type::nb_subel; iC++) {
        for (std::size_t iP = 0; iP < (std::size_t)plot_nodes.connectivity[iC].size(); iP++) {
          ss_connectivity << nb_points + plot_nodes.connectivity[iC](iP) << " " << std::flush;
        }  // for iP
        ss_connectivity << std::endl;
        nb_cells++;
      }  // for iC

      for (size_t iP = 0; iP < plot_nodes_type::nb_nodes; iP++) {
        auto xP = geo_map(plot_nodes.nodes[iP]);

        ss_points << xP.to_vector()(0) << " " << xP.to_vector()(1) << " ";

        Eigen::Matrix<scalar_type, element_type::cell_basis_type::dim, 1> u_P =
            Eigen::Matrix<scalar_type, element_type::cell_basis_type::dim, 1>::Zero();
        for (size_t i = 0; i < element_type::nb_local_cell_dofs; i++) {
          Eigen::Matrix<scalar_type, element_type::cell_basis_type::dim, 1> u_P_i;
          u_P_i << coeffs_T(i) * E.cell_basis().phi(i).phi(xP);
          u_P += u_P_i;
        }  // for i

        ss_points << u_P << std::endl;

        for (std::size_t i = 0; i < element_type::cell_basis_type::dim; i++) {
          ss_variable[i] << u_P(i) << " " << std::flush;
        }  // for i

        nb_points++;
      }  // for iP
    }    // for F
  }      // for iT

  for (std::size_t i = 0; i < element_type::cell_basis_type::dim; i++) {
    ss_variable[i] << "</DataArray>" << std::endl;
  }  // for i

  ss_points << "</DataArray>" << std::endl;
  ss_points << "</Points>" << std::endl;

  ss_connectivity << "</DataArray>" << std::endl;

  // Write output file, creating the directory if it does not exist
  assert_path(::boost::filesystem::path(file_name).parent_path());
  std::ofstream file(file_name, std::ios::out);
  file.flags(std::ios_base::scientific);

  file << "<?xml version=\"1.0\"?>" << std::endl;
  file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" "
          "compressor=\"vtkZLibDataCompressor\">"
       << std::endl;
  file << "<UnstructuredGrid>" << std::endl;

  file << "<Piece NumberOfPoints=\"" << nb_points << "\" NumberOfCells=\"" << nb_cells << "\">" << std::endl;

  file << "<PointData Scalars=\"" << std::flush;
  for (std::size_t i = 0; i < element_type::cell_basis_type::dim; i++) {
    file << variable_name << i << " " << std::flush;
  }  // for i
  file << "\">" << std::endl;

  for (std::size_t i = 0; i < element_type::cell_basis_type::dim; i++) {
    file << ss_variable[i].str() << std::endl;
  }  // for i
  file << "</PointData>" << std::endl;

  file << ss_points.str() << std::endl;

  file << "<Cells>" << std::endl;
  file << ss_connectivity.str() << std::endl;

  file << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
  for (std::size_t iC = 1; iC <= nb_cells; iC++) {
    file << iC * 3 << " " << std::flush;
  }
  file << std::endl;
  file << "</DataArray>" << std::endl;

  file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;
  for (std::size_t iC = 0; iC < nb_cells; iC++) {
    file << 5 << " " << std::flush;
  }
  file << std::endl;
  file << "</DataArray>" << std::endl;
  file << "</Cells>" << std::endl;

  file << "</Piece>" << std::endl;
  file << "</UnstructuredGrid>" << std::endl;
  file << "</VTKFile>" << std::endl;
}

template <size_t K, typename MESH_TYPE>
void post_processing(const std::string& file_name, const MESH_TYPE& Th,
                     const dynamic_vector<typename MESH_TYPE::scalar_type>& coeffs, const std::string& variable_name = "U") {
  post_processing<K>(file_name.c_str(), Th, coeffs, variable_name);
}

}  // namespace hho
