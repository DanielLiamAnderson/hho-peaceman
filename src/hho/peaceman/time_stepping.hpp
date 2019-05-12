// file  : time_stepping.hpp
// author: Daniel Anderson
//
// Encapsulate linear time-stepping methods for time-dependent
// PDEs
#pragma once

#include "hho/common.hpp"

namespace hho {

class time_stepping {
  
 public:
  time_stepping(const scalar_type delta_t, const size_t temporal_degree)
    : delta_t(delta_t), temporal_degree(temporal_degree) {
    // Do nothing
  }
  
  // Extrapolate two functions linearly as required by the particular time-stepping method
  // This could be a half-step for Crank-Nicholson, or a full step for BDF.
  //
  // Arguments:
  //  u0        : the function value at time-step n - 1
  //  u1        : the function value at time-step n      
  virtual cellwise_scalar_function extrapolate(const cellwise_scalar_function& u0, const cellwise_scalar_function& u1) = 0;
  
  // Compute the reaction term corresponding to the time discretisation
  //
  // Arguments:
  //  iT        : the cell id
  //  x         : a point in the cell iT
  virtual scalar_type reaction_term(const size_t iT, const point_type& x) = 0;
  
  // Compute the source term corresponding to the time discretisation
  //
  // Arguments:
  //  iT        : the cell id
  //  x         : a point in the cell iT
  virtual scalar_type source_term(const size_t iT, const point_type& x) = 0;
  
  // Add a new initial condition to use for time-stepping calculations
  //
  // Arguments:
  //  u         : the new initial condition
  virtual void update_initial_condition(const cellwise_scalar_function& u) = 0;
  
  // Extrapolate (if required) the solution given at Xt with respect to the previous solution Xp
  // to the correct time.
  //
  // Arguments:
  //  Xt        : the solution given by the time-step
  //  Xp        : the previous solution
  virtual dynamic_vector<scalar_type> post_processing(const dynamic_vector<scalar_type>& Xt, const dynamic_vector<scalar_type>& Xp) = 0;
  
  // Get the time at which to solve the system at time-step i_step
  //
  // Arguments:
  //  t0        : the initial time
  //  i_step    : the step number
  virtual scalar_type get_time(const scalar_type t0, const size_t i_step) = 0;
  
  // Accessors
  // Get the time-step size
  scalar_type get_delta_t() { return delta_t; }
  
  // Get the degree of the time-stepping scheme
  size_t get_temporal_degree() { return temporal_degree; }
  
  
  virtual ~time_stepping() { }
  
 protected:
  scalar_type delta_t;
  size_t temporal_degree; 
};

}  // namespace hho

