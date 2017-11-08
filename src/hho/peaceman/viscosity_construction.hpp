// file  : viscosity_construction.hpp
// author: Daniel Anderson
//
// Given the concentration, constructs the viscosity based on the mixing
// rule
//
//    mu(c) = mu(0)(1 + (M^{1/4} - 1)c)^{-4}
//
// The result is a cellwise scalar valued function of space
#pragma once

#include "hho/common.hpp"

namespace hho {

class viscosity_construction {
  
 public:
  viscosity_construction() {
    // Do nothing
  }
  
  // Reconstructs the viscosity using the given data
  //
  // Arguments:
  //  c       the concentration function
  //  m       the viscosity of the oil
  //  M       the mobility ratio of the fluids
  cellwise_scalar_function construct(const cellwise_scalar_function& c, const scalar_type m, const scalar_type M) {
    auto Mquart = std::pow(M, 0.25);
    auto viscosity = [=](const size_t iT, const point_type& x) {
      auto cx = std::min(1.0, std::max(0.0, c(iT, x)));
      return m * std::pow(1.0 + (Mquart - 1.0) * cx, -4.0);
    };
    return viscosity;
  }

};

}  // namespace hho


