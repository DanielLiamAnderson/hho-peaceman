// file  : test_peaceman.cpp
// author: Daniel Anderson
//
// Test the Peaceman solver.
//

#include <boost/program_options.hpp>
#include <boost/timer.hpp>

#include "hho/common.hpp"
#include "hho/integrator.hpp"
#include "hho/peaceman/peaceman_solver.hpp"
#include "hho/post_processing.hpp"

using namespace hho;

// The polynomial degree to use for the test
#ifdef DEGREE
constexpr int degree = DEGREE;
#else
constexpr int degree = 1;
#endif

// Output file directory
const std::string output_dir = "./output/peaceman/";

// Mesh file
const std::string mesh_dir = "./meshes/";
const std::string default_mesh = "scaled_2_2.typ1";

int main(int argc, const char* argv[]) {
  
  // ---------------------------------------------------------
  // Add program options
  // 
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Display this help message")
    ("mesh,m", boost::program_options::value<std::string>(), "Set the mesh")
    ("perm,p", boost::program_options::value<char>(), "Set the permeability tensor ('h' = homogeneous, 'd' = discontinuous)")
    ("time,t", boost::program_options::value<double>(), "Set the finish time")
    ("nsteps,n", boost::program_options::value<int>(), "Set the number of time steps")
    ("disc,d", boost::program_options::value<char>(), "Set the time discretisation ('c' = Crank-Nicholson, 'b' = Backward differentiation")
    ("order,o", boost::program_options::value<int>(), "Set the order of the time discretisation (for BDF time-stepping only)");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  // Display the help options
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  
  // Select the mesh
  std::string mesh_file = mesh_dir + (vm.count("mesh") ?
    vm["mesh"].as<std::string>() : default_mesh);
    
  // Select the permeability
  char perm_setting = vm.count("perm") ? vm["perm"].as<char>() : 'h';
  
  // Select the finish time
  double tf = vm.count("time") ? vm["time"].as<double>() : 3600.0;
  
  // Select the number of time steps
  int n_steps = vm.count("nsteps") ? vm["nsteps"].as<int>() : 200;
  
  // Select the time discretisation
  time_discretisation td = vm.count("disc") ?
    vm["disc"].as<char>() == 'c' ?
      time_discretisation::crank_nicholson :
      time_discretisation::backward_differentiation : 
    time_discretisation::crank_nicholson;
    
  // Select the order of the time discretisation
  int time_degree = vm.count("order,o") ? vm["order"].as<int>() : 1;
  
  // Files ---------------------------------------------------
  assert_directory(output_dir);
  std::string pressure_output_file = output_dir + "pressure.vtu";
  std::string concentration_output_file = output_dir + "concentration.vtu";
  
  //----------------------------------------------------------
  // Load mesh
  
  auto Th = std::make_shared<mesh_type>();
  fvca5_mesh_loader<double, 2> loader;
  loader.read_mesh(mesh_file);
  loader.populate_mesh(*Th);
  
  std::cout << "Mesh statistics::" << std::endl;
  std::cout << "\tnum_nodes = " << Th->points_size() << std::endl;
  std::cout << "\tnum_cells = " << Th->cells_size() << std::endl;
  std::cout << "\tnum_faces = " << Th->faces_size() << std::endl;
  std::cout << "\tnum_internal_faces = " << Th->internal_faces_size() << std::endl;
  std::cout << "\tnum_boundary_faces = " << Th->boundary_faces_size() << std::endl;
  std::cout << "\tmeshsize = " << mesh_h(*Th) << std::endl;
  
  //----------------------------------------------------------
  // Parameters
  
  peaceman_solver_parameters params;
  params.t0 = 0.0;
  params.t1 = tf;
  params.n_steps = n_steps;
  params.temporal_degree = time_degree;
  params.time_derivative = td;
  
  //----------------------------------------------------------
  // Data
  
  peaceman_model_data data;
  
  // The injection and production wells
  data.injection_well_loc = {1000.0, 1000.0};
  data.production_well_loc = {0.0, 0.0};
  data.flow_rate = [](const scalar_type t) { return 30.0; };
  data.injected_concentration = [](const scalar_type t) { return 1.0; };
  
  // Constant porosity
  data.porosity = [](const size_t iT, const point_type& x) { return 0.1; };
  
  // Homogeneous permeability or discontinuous permeability
  auto K = Eigen::Matrix<scalar_type, 2, 2>::Identity();
  if (perm_setting == 'h') {
    data.permeability = [=](const size_t iT, const point_type& x) {
      return 80.0*K;
    };
  }
  else {
    data.permeability = [=](const size_t iT, const point_type& x) {
      return 5 * K + (x.x() < 200 || x.x() > 800 || x.y() < 200 || x.y() > 800 || std::abs(x.x() - 500) < 100 || std::abs(x.y() - 500) < 100) * 75.0 * K;
    };
  }
    
  // Viscosity
  data.oil_viscosity = 1.0;
  data.mobility_ratio = 41.0;
  
  // Diffusion coefficients
  data.dm = 0.0;
  data.dl = 50.0;
  data.dt = 5.0;
  
  // Initial condition c(0, x) = 0
  data.initial_condition = [](const size_t iT, const point_type& x) { return 0.0; };

  //----------------------------------------------------------
  // Create solver and solve the problem
  
  std::cout << "Initialising the solver..." << std::endl;
  
  peaceman_solver<degree> solver(Th);
  integrator<degree> quad(Th);
  
  std::cout << "Solving the system..." << std::endl;
  boost::timer timer;
  
  auto sol = solver.solve(data, params, true);
  
  std::cout << "Successfully solved the system in " << timer.elapsed() << "sec" << std::endl;
  std::cout << "Saving results..." << std::endl;
  
  // Analyse the results
  auto total_volume = quad.integrate(data.porosity);
  auto extracted_volume = quad.l2_inner_product(sol.second, data.porosity);
  auto percent = extracted_volume / total_volume * 100.0;
  auto l2norm = quad.l2_norm(sol.second);
  
  std::cout << "Total volume of oil = " << total_volume << std::endl;
  std::cout << "Extracted volume of oil = " << extracted_volume << std::endl;
  std::cout << "Percentage of oil extracted = " << percent << std::endl;
  std::cout << "L^2 norm of the concentration = " << l2norm << std::endl;
  
  // Save the output
  post_processing<2*degree>(pressure_output_file, *Th, sol.first);
  post_processing<degree>(concentration_output_file, *Th, sol.second);

  std::cout << "Plots saved to " << pressure_output_file << " and " << concentration_output_file << std::endl;
}
