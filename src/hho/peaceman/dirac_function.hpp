// file  : dirac_function.hpp
// author: Daniel Anderson
//
// Constructs source terms represented by Dirac masses, approximated
// by Piecewise constant functions on the mesh.
//
#pragma once

#include "hho/common.hpp"

namespace hho {

// Contains the data required to represent a set of Dirac mass source terms
class dirac_function {

 public:
  
  dirac_function() {
    // Do nothing
  }

  // Create a source term represented by Dirac masses at the given locations with constant flow rate
  cellwise_scalar_function_of_time create(const mesh_type& Th, const point_type& location, const scalar_type flow) {
    return create(Th, location, [flow](const scalar_type t) { return flow; });
  }

  // Create a source term represented by Dirac masses at the given locations with variable flow rate
  cellwise_scalar_function_of_time create(const mesh_type& Th, const point_type& location, const scalar_function_of_time flow) {
  
    // ---------------------------------------------------------------------------------
    // Record the total measure of the cells containing the mass
    
    auto m_T = 0.0;
    auto has_well = std::make_shared<std::vector<bool>>(Th.cells_size());
    
    // For each cell T in T_h
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = std::distance(Th.cells_begin(), it_T);
      if (contains_point(Th, T, location)) {
        m_T += measure(Th, T);
        has_well->at(iT) = true;
      }
    }
    
    // ---------------------------------------------------------------------------------
    // Assemble the function
    
    auto source_term = [=](const scalar_type t) {
      return [=](const size_t iT, const point_type& x) {
        if (has_well->at(iT)) {
          return flow(t) / m_T;
        } else {
          return 0.0;
        }
      };
    };
    
    return source_term;
  }

};


}  // namespace hho
