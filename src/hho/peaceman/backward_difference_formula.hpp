// file  : backward_difference_formula.hpp
// author: Daniel Anderson
//
// Encapsulates the information required to perform BDF time
// stepping.
#pragma once

#include <deque>

#include "hho/common.hpp"

#include "time_stepping.hpp"

namespace hho {

class backward_difference_formula : public time_stepping {

  // Coefficients for backward difference formulas
  const std::vector<std::vector<double>> coeffs = {
    // Order 1 BDF
    {1.0, -1.0},
    
    // Order 2 BDF
    {3.0/2.0, -4.0/2.0, 1.0/2.0},
    
    // Order 3 BDF
    {11.0/6.0, -18.0/6.0, 9.0/6.0, -2.0/6.0},
    
    // Order 4 BDF
    {25.0/12.0, -48.0/12.0, 36.0/12.0, -16.0/12.0, 3.0/12.0},
    
    // Order 5 BDF
    {137.0/60.0, -300.0/60.0, 300.0/60.0, -200.0/60.0, 75.0/60.0, -12.0/60.0},
    
    // Order 6 BDF
    {147.0/60.0, -360.0/60.0, 450.0/60.0, -400.0/60.0, 225.0/60.0, -72.0/60.0, 10.0/60.0},
  };
  
 public:
  
  // Create a backward differencing handler using the k'th BDF formula
  backward_difference_formula(const scalar_type delta_t, const size_t k)
    : time_stepping(delta_t, k) {
    // Do nothing
  }

  // Do a time step
  virtual scalar_type get_time(const scalar_type t0, const size_t i_step) {
    return t0 + i_step * get_delta_t();
  }

  // Extrapolate two functions by a time-step
  virtual cellwise_scalar_function extrapolate(const cellwise_scalar_function& u0, const cellwise_scalar_function& u1) {
    return [=](const size_t iT, const point_type& x) {
      return 2.0 * u1(iT, x) - 1.0 * u0(iT, x);
    };
  }

  // Add an initial condition to the system
  virtual void update_initial_condition(const cellwise_scalar_function& ic) {
    u.push_back(ic);
    while (u.size() > get_temporal_degree() + 1) {
      u.pop_front();
    }
  }

  // Compute the reaction term of the backward difference formula
  virtual scalar_type reaction_term(const size_t iT, const point_type& x) {
    return coeffs[get_temporal_degree()][0] / get_delta_t();
  }
  
  // Compute the source term of the backward difference formula
  virtual scalar_type source_term(const size_t iT, const point_type& x) {
    assert(!u.empty());
    scalar_type res = 0.0;
    for (size_t i = 1; i <= get_temporal_degree() + 1; i++) {
      size_t index = (u.size() >= i ? u.size() - i : 0);
      res += coeffs[get_temporal_degree()][i] / get_delta_t() * u[index](iT, x);
    }
    return res;
  }
  
  // No extrapolation required
  virtual dynamic_vector<scalar_type> post_processing(const dynamic_vector<scalar_type>& Xt, const dynamic_vector<scalar_type>& Xp) {
    return Xt;
  }

 private:
  std::deque<cellwise_scalar_function> u;       // Initial conditions of the scheme
};

}  // namespace hho


