// file: test_oil_volume.cpp
// author: Daniel Anderson
//
// Tests the estimate on the volume of recovered oil to observe
// if there seems to be any convergence.
//

#include <boost/program_options.hpp>

#include "hho/common.hpp"
#include "hho/peaceman/peaceman_solver.hpp"

#include "test_base.hpp"

using namespace hho;

// Meshes scaled to 1000 * 1000 for reservoir simulations
const std::vector<std::vector<std::string>> scaled_meshes = {
  // Triangular mesh
  {"./meshes/scaled_1_1.typ1",
   "./meshes/scaled_1_2.typ1",
   "./meshes/scaled_1_3.typ1",
   "./meshes/scaled_1_4.typ1",
   "./meshes/scaled_1_5.typ1"},
  // Square mesh
  {"./meshes/scaled_2_1.typ1",
   "./meshes/scaled_2_2.typ1",
   "./meshes/scaled_2_3.typ1",
   "./meshes/scaled_2_4.typ1",
   "./meshes/scaled_2_5.typ1"},
  // Kershaw meshes
  {"./meshes/scaled_4_1_1.typ1",
   "./meshes/scaled_4_1_2.typ1",
   "./meshes/scaled_4_1_3.typ1",
   "./meshes/scaled_4_1_4.typ1",
   "./meshes/scaled_4_1_5.typ1"},
  // Skewed hexagonal mesh
  {"./meshes/scaled_pi6_1.typ1",
   "./meshes/scaled_pi6_2.typ1",
   "./meshes/scaled_pi6_3.typ1",
   "./meshes/scaled_pi6_4.typ1",
   "./meshes/scaled_pi6_5.typ1"}
};

// Mesh names
std::vector<std::string> mesh_names = {
  "Triangular meshes",
  "Square meshes",
  "Kershaw meshes",
  "Tilted hexagonal meshes"
};

// Test data-set for the poisson-dirichlet equation
struct test_data : public generic_test_data {
  peaceman_solver_parameters params;
  peaceman_model_data data;
};

// Tester class for the advection-diffusion-reaction equation
class tester : public generic_tester {
  
  // Override the generic test functions (see test_base.hpp)
  OVERRIDE_TESTS;
  
  // Tests the solver and return the error
  template<size_t degree>
  test_result test(mesh_ptr& mesh, data_ptr& g_data) {
    
    std::cout << "\t\tTesting degree " << degree << "... " << std::flush;
    
    // Cast the data to the correct type
    const std::shared_ptr<test_data> peaceman_data = std::dynamic_pointer_cast<test_data>(g_data);
    const test_data& t_data = *peaceman_data;
    
    // Create solvers / tools
    peaceman_solver<degree> solver(mesh);
    integrator<degree> quad(mesh);

    // Solve and measure volume
    auto sol = solver.solve(t_data.data, t_data.params);
    
    scalar_type total_volume = quad.integrate(t_data.data.porosity);
    scalar_type extracted_volume = quad.l2_inner_product(sol.second, t_data.data.porosity);
    scalar_type percent = extracted_volume / total_volume * 100.0;
    
    // Error computation
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " volume = " << percent << "% " << std::endl;
    
    return std::make_pair(percent, std::move(sol.second));
  }
};

int main(int argc, char* argv[]) {
  
  // --------------------------------------------------------------------------
  // Add program options
  // 
  // The ability to run the tests on a particular set of meshes
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Display this help message")
    ("mesh,m", boost::program_options::value<size_t>(), "Set the mesh family (if not set, will run ALL)");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  // Display the help options
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }
  
  // Create an instance of the tester and create tests
  tester T;
  
  // Use only one of the mesh sets
  if (vm.count("mesh")) {
     size_t mesh_id = vm["mesh"].as<size_t>();
     T.set_meshes({scaled_meshes[mesh_id]}, {mesh_names[mesh_id]});
  }
  else {
     T.set_meshes(scaled_meshes, mesh_names);
  }
  
  // ----------------------------------------------------------------
  //                          Standard test
  // ----------------------------------------------------------------
  {
    test_data* t_data = new test_data();
    auto& data = t_data->data;
    auto& params = t_data->params;
    
    //----------------------------------------------------------
    // Parameters
    
    params.t0 = 0.0;
    params.t1 = 3600.0;
    params.n_steps = 500;
    params.temporal_degree = 3;
    params.time_derivative = time_discretisation::backward_differentiation;
    
    //----------------------------------------------------------
    // Data
    
    // The injection and production wells
    data.injection_well_loc = {1000.0, 1000.0};
    data.production_well_loc = {0.0, 0.0};
    data.flow_rate = [](const scalar_type t) { return 30.0; };
    data.injected_concentration = [](const scalar_type t) { return 1.0; };
    
    // Constant porosity
    data.porosity = [](const size_t iT, const point_type& x) { return 0.1; };
    
    // Homogeneous permeability
    auto K = Eigen::Matrix<scalar_type, 2, 2>::Identity();
    data.permeability = [=](const size_t iT, const point_type& x) {
      return 80.0*K;
    };
    
    // Viscosity
    data.oil_viscosity = 1.0;
    data.mobility_ratio = 41.0;
    
    // Diffusion coefficients
    data.dm = 0.0;
    data.dl = 50.0;
    data.dt = 5.0;
    
    // Initial condition c(0, x) = 0
    data.initial_condition = [](const size_t iT, const point_type& x) { return 0.0; };
    
    T.add_test("Recovery Volume Peaceman standard test", std::shared_ptr<generic_test_data>(t_data));
  }
  
  // Run the tests
  T.run_tests();
  
  return 0;
}

