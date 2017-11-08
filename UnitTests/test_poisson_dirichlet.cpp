// file: test_poisson_neumann.cpp
// author: Daniel Anderson
//
// Tests for the HHO scheme for Poisson with Dirichlet boundary conditions
//

#include "hho/common.hpp"
#include "hho/poisson_dirichlet_solver.hpp"

#include "test_base.hpp"

using namespace hho;

const auto _pi = acos(-1.0);

// Test data-set for the poisson-dirichlet equation
struct test_data : public generic_test_data {
  scalar_function_of_space exact_solution;
  cellwise_scalar_function boundary_condition;
  cellwise_scalar_function source_term;
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
    const std::shared_ptr<test_data> pn_data = std::dynamic_pointer_cast<test_data>(g_data);
    const test_data& data = *pn_data;
    
    // Create solvers / tools
    poisson_dirichlet_solver<degree> solver(mesh);
    global_interpolator<degree> interpolator(solver);
    integrator<degree> quad(solver);
    
    // Solve the scheme and compare with exact solution
    auto Xh = solver.solve(data.source_term, data.boundary_condition);
    auto XTF = interpolator.global_interpolant(data.exact_solution);
    auto diff = Xh - XTF;
    
    // Error computation
    auto error = quad.l2_norm(diff);
    std::cout << " error = " << error << ". " << std::endl;
    
    return std::make_pair(error, std::move(Xh));
  }
};

int main() {

  // Create an instance of the tester and create tests
  tester T;

  // ----------------------------------------------------------------
  //                     Zero boundary conditions
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- sin(pi x) sin(pi y)
    data->exact_solution = [](const point_type& x) { \
      return sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    // Dirichlet boundary value
    data->boundary_condition = [](const size_t iT, const point_type& x) {
      return 0.0;
    };
    
    // Diffusive source term
    data->source_term = [](const size_t iT, const point_type& x) {
      return 2.0 * (_pi * _pi) * sin(_pi * x.x()) * sin(_pi * x.y());        
    }; 
    
    T.add_test("Poisson Dirichlet zero boundary", std::shared_ptr<generic_test_data>(data));
  }

  // ----------------------------------------------------------------
  //                     Non-zero boundary conditions
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- cos(pi x) cos(pi y)
    data->exact_solution = [](const point_type& x) {
      return cos(_pi * x.x()) * cos(_pi * x.y());
    };
    
    // Dirichlet boundary value
    data->boundary_condition = [](const size_t iT, const point_type& x) {
      return cos(_pi * x.x()) * cos(_pi * x.y());
    };
    
    // Diffusive source term
    data->source_term = [](const size_t iT, const point_type& x) {
      return 2.0 * (_pi * _pi) * cos(_pi * x.x()) * cos(_pi * x.y());        
    }; 
    
    T.add_test("Poisson Dirichlet nonzero boundary", std::shared_ptr<generic_test_data>(data));
  }
  
  // Run the tests
  T.run_tests();
  
  return 0;
}

