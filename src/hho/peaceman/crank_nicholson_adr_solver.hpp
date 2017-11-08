// file: crank_nicholson_adr_solver.hpp
// author: Daniel Anderson
//
// Solves a time-dependent advection-diffusion system using
// Crank-Nicholson time-stepping.
//
#pragma once

#include <vector>

#include "hho/adr_solver.hpp"
#include "hho/common.hpp"
#include "hho/global_interpolator.hpp"

#include "crank_nicholson.hpp"
#include "function_reconstruction.hpp"

namespace hho {

template<size_t spatial_degree>
class crank_nicholson_adr_solver {

 public:
  // Initialise a solver with the given mesh
  crank_nicholson_adr_solver(const std::shared_ptr<mesh_type> mesh)
    : solver(mesh), reconstructor(solver), interpolator(solver) {
    // Do nothing
  }
 
  // Initialise a solver with the given mesh
  crank_nicholson_adr_solver(const std::string& mesh)
    : solver(mesh), reconstructor(solver), interpolator(solver) {
    // Do nothing
  }
 
  // Access the mesh data
  const mesh_type& get_mesh() { return solver.get_mesh(); }
 
  // Solves the time dependent Advection-Diffusion-Reaction equation using the given data and returns the HHO solution
  // at each time step in order.
  //
  // Arguments:
  //  t0        the initial time value
  //  t1        the final time value
  //  n_steps   the number of time steps to take
  //  source    the source term f
  //  boundary  the value of the directional derrivative of the solution at the given point, in the given direction
  //  diffusion_tensor  the diffusion tensor as a function of space 
  //  velocity  the velocity vector as a function of space
  //  reaction  the reaction terms as a function of space
  //  initial_condition  the initial conditions of the system
  //  store_all true to store and return all time-steps of the solution
  //
  std::vector<dynamic_vector<scalar_type>> solve(const scalar_type t0, const scalar_type t1, const size_t n_steps,
    const cellwise_scalar_function_of_time& source, const cellwise_flux_function_of_time& boundary, 
    const cellwise_tensor_function_of_time& diffusion_tensor, const cellwise_vector_function_of_time& velocity,
    const cellwise_flux_function_of_time& flux, const cellwise_scalar_function_of_time& reaction,
    cellwise_scalar_function& initial_condition, bool store_all = false) {
    
    // Compute parameters
    auto delta_t = (t1 - t0) / n_steps;
    
    // Insert initial condition
    crank_nicholson cn(delta_t, 1);
    cn.update_initial_condition(initial_condition);
    
    // Results
    std::vector<dynamic_vector<scalar_type>> results;
    results.reserve(n_steps + store_all);
    
    // Compute initial state
    auto Xp = interpolator.global_interpolant(initial_condition);
    if (store_all) {
      results.push_back(Xp);
    }
    
    // Perform each time step
    for (size_t i = 1; i <= n_steps; i++) {
      // Make a half-step because Crank-Nicholson
      auto t = cn.get_time(t0, i);
      
      // Evaluate problem data at time t      
      auto mu_t = reaction(t);
      auto f_t = source(t);
    
      // Factor in Crank-Nicholson terms
      auto reaction_t = [&](const size_t iT, const point_type& x) {
        return mu_t(iT, x) + cn.reaction_term(iT, x);
      };
      auto source_t = [&](const size_t iT, const point_type& x) {
        return f_t(iT, x) - cn.source_term(iT, x);
      };
    
      // Solve the corresponding stationary problem
      auto Xt = solver.solve(source_t, boundary(t), diffusion_tensor(t), velocity(t), flux(t), reaction_t);
    
      // Extrapolate u^{n+0.5} to get back u^{n+1}
      Xp = cn.post_processing(Xt, Xp);
    
      // Store the results
      if (store_all || i == n_steps) {
        results.push_back(Xp);
      }
      cn.update_initial_condition(reconstructor.reconstruct(Xp));
    }
    
    return results;
  }

 private:
  adr_solver<spatial_degree> solver;
  function_reconstruction<spatial_degree> reconstructor;
  global_interpolator<spatial_degree> interpolator;
};

}  // namespace hho
