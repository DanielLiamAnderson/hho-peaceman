// file: test_function_reconstruction.cpp
// author: Daniel Anderson
//
// Tests the function reconstruction.
//

#include "hho/common.hpp"
#include "hho/peaceman/function_reconstruction.hpp"

#include "test_base.hpp"

using namespace hho;

const auto _pi = acos(-1.0);

// Test data-set for the poisson-dirichlet equation
struct test_data : public generic_test_data {
  scalar_function_of_space exact_solution;
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
    const std::shared_ptr<test_data> re_data = std::dynamic_pointer_cast<test_data>(g_data);
    const test_data& data = *re_data;
    
    // Create solvers / tools
    function_reconstruction<degree> reconstructor(mesh);
    global_interpolator<degree> interpolator(reconstructor);
 
    // Interpolate the exact solution
    auto XTF = interpolator.global_interpolant(data.exact_solution);
    
    // Reconstruct the Darcy velocity
    auto reconstruction = reconstructor.reconstruct(XTF);
   
    // Compute the error
    const auto& Th = *mesh;
    auto error = 0.0;
    
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = std::distance(Th.cells_begin(), it_T);
      const auto& E = reconstructor.get_local_elements()[iT];
      auto FT = faces(Th, T); 

      // Compute the error in this subelement
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        const auto iF_loc = std::distance(FT.begin(), it_F);
        const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);

        for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
          const auto& xQN = qr_PTF.first[iQN];
          const auto& wQN = qr_PTF.second[iQN];
          
          // Compute the exact value and reconstructed value
          auto exact_value_iqn = data.exact_solution(xQN);
          auto reconstructed_value_iqn = reconstruction(iT, xQN);
          auto diff = exact_value_iqn - reconstructed_value_iqn;

          error += wQN * (diff * diff);
        }
      }
    }
    
    error = std::pow(error, 0.5);
    
    // Error computation
    std::cout << " error = " << error << ". " << std::endl;
    
    return std::make_pair(error, std::move(XTF));
  }
};

int main() {

  // Create an instance of the tester and create tests
  tester T;
  
  // ----------------------------------------------------------------
  //                Function reconstruction test
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- sin(pi x) sin(pi y)
    data->exact_solution = [](const point_type& x) {
      return sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    T.add_test("Function reconstruction test", std::shared_ptr<generic_test_data>(data));
  }
  
  // Run the tests
  T.run_tests();
  
  return 0;
}

