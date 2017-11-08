// file: peaceman_solver.hpp
// author: Daniel Anderson
//
// Solves the Peaceman Model.
//
#pragma once

#include <boost/timer.hpp>

#include <deque>
#include <vector>

#include "hho/adr_solver.hpp"
#include "hho/common.hpp"
#include "hho/diffusion_neumann_solver.hpp"
#include "hho/global_interpolator.hpp"

#include "backward_difference_formula.hpp"
#include "crank_nicholson.hpp"
#include "darcy_velocity_reconstruction.hpp"
#include "diffusion_dispersion_tensor_construction.hpp"
#include "dirac_function.hpp"
#include "function_reconstruction.hpp"
#include "time_stepping.hpp"
#include "viscosity_construction.hpp"

namespace hho {

// ----------------------------------------------------------------------------------------
//                                Data and parameters
// ----------------------------------------------------------------------------------------

// Data for an instance of a reservoir problem modelled by the Peaceman Model
//
// injection_well_loc / production_well_loc :     The locations of the wells
// flow_rate :                                    The flow rate of the wells
// injected_concentration :                       The concentration of solvent in the injected fluid
// porosity :                                     The porosity of the porous media, phi
// permeability :                                 The permeability tensor, K
// oil_viscosity, mobility_ratio :                The viscosity of the oil and the mobility ratio of the fluids
// dm, dl, dt :                                   The molecular, longitudinal and tranverse diffusivity
// initial_condition :                            The initial concentration in the well
//
struct peaceman_model_data {
  point_type injection_well_loc, production_well_loc;
  scalar_function_of_time flow_rate;
  scalar_function_of_time injected_concentration;
  cellwise_scalar_function porosity;
  cellwise_tensor_function permeability;
  scalar_type oil_viscosity, mobility_ratio;
  scalar_type dm, dl, dt;
  cellwise_scalar_function initial_condition;
};

// Options for the time discretisation
enum class time_discretisation {
  backward_differentiation,
  crank_nicholson
};

// Factory method for creating time stepping objects
std::unique_ptr<time_stepping> get_time_derrivative(const time_discretisation& type,
  const scalar_type delta_t, const size_t temporal_degree) {
  if (type == time_discretisation::backward_differentiation) {
    return std::make_unique<backward_difference_formula>(delta_t, temporal_degree);
  } else {
    return std::make_unique<crank_nicholson>(delta_t, temporal_degree);
  }
}

// Parameters for the Peaceman model solver
//
// t0, t1 :         The start and end times of the time interval in question
// time_stepping :  The kind of time discretisation to use
// n_steps :        The number of time steps to perform
// temporal_degree: The degree of accuracy to use in the temporal approximations
//
struct peaceman_solver_parameters {
  scalar_type t0, t1;
  time_discretisation time_derivative;
  size_t n_steps, temporal_degree;
};

// ----------------------------------------------------------------------------------------
//                                Solver implementation
// ----------------------------------------------------------------------------------------

// Solver for the peaceman model.
//
// Usage:
//  
//    peaceman_solver<spatial_degree> solver(mesh_ptr);
//    auto Ch = solver.solve(data, params);
//
// where data is an instance of peaceman_model_data and params
// is an instance of peaceman_solver_parameters
//
// Parameters:
//  spatial_degree    the polynomial degree of the concentration estimate
template<size_t spatial_degree>
class peaceman_solver {

  // Typedefs
  typedef typename adr_solver<spatial_degree>::element_type element_type;

 public:
  // Initialise a solver with the given mesh
  peaceman_solver(std::shared_ptr<mesh_type> mesh)
    : pressure_solver(mesh),
      velocity_reconstructor(pressure_solver),
      concentration_solver(mesh),
      dirac_mass(),
      function_reconstructor(concentration_solver),
      diffusion_dispersion_tensor_constructor(),
      interpolator(concentration_solver),
      viscosity_constructor()
  {
    // Do nothing
  }
 
  // Initialise a solver with the given mesh from file
  peaceman_solver(const std::string& mesh)
    : pressure_solver(mesh),
      velocity_reconstructor(pressure_solver),
      concentration_solver(mesh),
      dirac_mass(),
      function_reconstructor(concentration_solver),
      diffusion_dispersion_tensor_constructor(),
      interpolator(concentration_solver),
      viscosity_constructor()
  {
    // Do nothing
  }
 
  // Internal accessors to shared data
  const mesh_type& get_mesh() { return concentration_solver.get_mesh(); }
  const std::vector<element_type>& get_local_elements() { return concentration_solver.get_local_elements(); }
  const std::map<mesh_type::face, size_t>& get_gfn() { return concentration_solver.get_gfn(); }
  const std::vector<dynamic_vector<size_t>>& get_dof_map() { return concentration_solver.get_dof_map(); }
 
  // Solves the Peaceman model using the given data and parameters and
  // returns the final HHO solution.
  //
  // Arguments:
  //  data          the data for the model
  //  parameters    the parameters for the solver
  //
  // Returns:
  //  a pair containing the HHO solution to the final pressure
  //  and final concentration respectively
  //
  std::pair<dynamic_vector<scalar_type>, dynamic_vector<scalar_type>> solve(const peaceman_model_data& data,
    const peaceman_solver_parameters& params, const bool loud = false) {
    
    // ------------------------------------------------------------------------
    //                          Initialisation
    // ------------------------------------------------------------------------
    
    // Initialise time stepping
    auto delta_t = (params.t1 - params.t0) / params.n_steps;
    auto dc_dt = get_time_derrivative(params.time_derivative, delta_t, params.temporal_degree);
    dc_dt->update_initial_condition(data.initial_condition);
    
    // Keep two concentration values for linear extrapolation
    std::deque<cellwise_scalar_function> concentration_values;        
    concentration_values.push_back(data.initial_condition);
    concentration_values.push_back(data.initial_condition);
    
    // Record initial value
    auto Cp = interpolator.global_interpolant(data.initial_condition);
    
    // Create Dirac mass source terms
    auto q_plus = dirac_mass.create(get_mesh(), data.injection_well_loc, data.flow_rate);
    auto q_minus = dirac_mass.create(get_mesh(), data.production_well_loc, data.flow_rate);
 
    // Create homogeneous no-flow Neumann boundary conditions
    auto boundary_condition = [&](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      return 0.0;
    };
 
    // ------------------------------------------------------------------------
    //                      Perform time stepping
    // ------------------------------------------------------------------------
    
    boost::timer timer;
    for (size_t i_step = 1; i_step <= params.n_steps; i_step++) {
    
      if (loud) {
        std::cout << "Time step: " << i_step << '/' << params.n_steps << "\r" << std::flush;
      }
    
      // Evaluate current time step
      auto t = dc_dt->get_time(params.t0, i_step);
      
      // ----------------------------------------------------------------------
      //                 Solve the pressure equation for p^i
    
      // Evaluate source term at time t
      auto q_plus_t = q_plus(t);
      auto q_minus_t = q_minus(t);
      
      // Create pressure equation source term
      auto pressure_source_term = [&](const size_t iT, const point_type& x) {
        return q_plus_t(iT, x) - q_minus_t(iT, x);
      };
      
      // Extrapolate the concentration in order to estimate the viscosity
      auto c_extrapolated = dc_dt->extrapolate(concentration_values[0], concentration_values[1]);
      auto viscosity = viscosity_constructor.construct(c_extrapolated, data.oil_viscosity, data.mobility_ratio);
      
      // Create the diffusion tensor kappa = K(x) / mu(c)
      auto pressure_kappa = [&](const size_t iT, const point_type& x) {
        return data.permeability(iT, x) / viscosity(iT, x);
      };
      
      // Solve for the pressure
      auto Ph = pressure_solver.solve(pressure_source_term, boundary_condition, pressure_kappa);
      
      
      // ----------------------------------------------------------------------
      //                 Solve the concentration equation for c^i
    
      // Create the source term (including the BDF terms)
      auto concentration_source_term = [&](const size_t iT, const point_type& x) {
        return data.injected_concentration(t) * q_plus_t(iT, x) - data.porosity(iT, x) * dc_dt->source_term(iT, x);
      };
    
      // Create the reaction term (including the BDF term)
      auto concentration_reaction_term = [&](const size_t iT, const point_type& x) {
        return q_minus_t(iT, x) + data.porosity(iT, x) * dc_dt->reaction_term(iT, x);
      };
    
      // Reconstruct the numerical Darcy velocity and the fluxes
      auto darcy_velocity = velocity_reconstructor.reconstruct(Ph, pressure_kappa);
      auto& U_T = darcy_velocity.first;
      auto& U_TF = darcy_velocity.second;
      
      // Construct the diffusion-dispersion tensor D from the Darcy velocity
      auto D = diffusion_dispersion_tensor_constructor.construct(U_T, data.porosity, data.dm, data.dt, data.dl);
    
      // Solve for the concentration
      auto Ct = concentration_solver.solve(concentration_source_term, boundary_condition, D, U_T, U_TF,
        concentration_reaction_term);
    
      // Extrapolate to correct for half time-stepping methods
      Cp = dc_dt->post_processing(Ct, Cp);
      
      // ----------------------------------------------------------------------
      //                          Post processing
      
      // If we're at the last time step -- return the solution
      if (i_step == params.n_steps) {
        avg_time_step = timer.elapsed() / params.n_steps;
        return std::make_pair(std::move(Ph), std::move(Cp));
      }
      // Otherwise -- reconstruct the concentration to use for the next time step
      else {
        auto Ch = function_reconstructor.reconstruct(Cp);
        dc_dt->update_initial_condition(Ch);
        concentration_values.push_back(Ch);
        concentration_values.pop_front();
      }
    }  // for i_step
    
    return std::make_pair(dynamic_vector<scalar_type>(), dynamic_vector<scalar_type>());
  }

  // Accessors
  scalar_type get_avg_time_step() { return avg_time_step; }

 private:
  // Tools and solvers
  // Pressure and Darcy velocity (Solved at degree 2k)
  diffusion_neumann_solver<2*spatial_degree> pressure_solver;
  darcy_velocity_reconstruction<2*spatial_degree> velocity_reconstructor;
  
  // Concentration (Solved at degree k)
  adr_solver<spatial_degree> concentration_solver;

  // Parameters and reconstructions
  dirac_function dirac_mass;
  function_reconstruction<spatial_degree> function_reconstructor;
  diffusion_dispersion_tensor_construction diffusion_dispersion_tensor_constructor;
  global_interpolator<spatial_degree> interpolator;
  viscosity_construction viscosity_constructor;
  
  // Time measurement
  scalar_type avg_time_step;
};

}  // namespace hho
