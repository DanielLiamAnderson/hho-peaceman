// file  : diffusion_dispersion_tensor_construction.hpp
// author: Daniel Anderson
//
// Constructs the diffusion-disperson tensor from the given
// Darcy velocity and porosity.

#pragma once

#include "hho/common.hpp"

namespace hho {

class diffusion_dispersion_tensor_construction {
  
 public:
  diffusion_dispersion_tensor_construction() {
    // Do nothing
  }
  
  // Construct the diffusion-dispersion tensor D from the given parameters
  //
  // Arguments:
  //  velocity    the Darcy velocity of the fluid
  //  phi         the porosity of the porous media
  //  dm          the molecular diffusion coefficient
  //  dt          the transverse diffusion coefficient
  //  dl          the longitudinal diffusion coefficient
  cellwise_tensor_function construct(const cellwise_vector_function& U, const cellwise_scalar_function& phi,
      scalar_type dm, scalar_type dt, scalar_type dl) {
    
    auto D = [=](const size_t iT, const point_type& x) -> tensor_type {
      auto U_x = U(iT, x);
      auto phi_x = phi(iT, x);
      auto eye = tensor_type::Identity();
      
      // Special case when U is zero or close to zero
      if (U_x.norm() < 1e-8) {
        return phi_x * dm * eye;
      }
      else {
        auto E = (U_x * U_x.transpose()) / U_x.norm();   // Compute outer product
        return phi_x * (dm * eye + dl * E + dt * (eye - E));
      }
    };
    return D;
  }
  
};

}  // namespace hho

