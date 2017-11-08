// file  : crank_nicholson.hpp
// author: Daniel Anderson
//
// Encapsulates the information required to perform 
// time stepping with Crank-Nicholson
#pragma once

#include "hho/common.hpp"

#include "time_stepping.hpp"

namespace hho {

class crank_nicholson : public time_stepping {
 public:
  
  // Create a Crank-Nicholson time-stepping formula
  crank_nicholson(const scalar_type delta_t, const size_t temporal_degree)
    : time_stepping(delta_t, 1) {
    // Do nothing
  }

  // Do a half time step
  virtual scalar_type get_time(const scalar_type t0, const size_t i_step) {
    return t0 + (i_step - 0.5) * delta_t;
  }

  // Extrapolate two functions by a half time-step
  virtual cellwise_scalar_function extrapolate(const cellwise_scalar_function& u0, const cellwise_scalar_function& u1) {
    return [=](const size_t iT, const point_type& x) {
      return 1.5 * u1(iT, x) - 0.5 * u0(iT, x);
    };
  }

  // Set the initial condition for the system
  virtual void update_initial_condition(const cellwise_scalar_function& ic) {
    u = ic;
  }

  // Compute the reaction term of the Crank-Nicholson formula
  virtual scalar_type reaction_term(const size_t iT, const point_type& x) {
    return 2.0 / get_delta_t();
  }
  
  // Compute the source term of the Crank-Nicholson formula
  virtual scalar_type source_term(const size_t iT, const point_type& x) {
    return - 2.0 * u(iT, x) / get_delta_t();
  }
  
  // Extrapolate the solution linearly
  virtual dynamic_vector<scalar_type> post_processing(const dynamic_vector<scalar_type>& Xt, const dynamic_vector<scalar_type>& Xp) {
    return 2 * Xt - Xp;
  }

 private:
  cellwise_scalar_function u;                   // Previous solution
};

}  // namespace hho


