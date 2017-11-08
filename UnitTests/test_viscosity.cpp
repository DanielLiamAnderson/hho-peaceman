// file: test_viscosity.cpp
// author: Daniel Anderson
//
// Tests the viscosity construction.
//

#include "hho/common.hpp"
#include "hho/peaceman/function_reconstruction.hpp"
#include "hho/peaceman/viscosity_construction.hpp"

#include "test_base.hpp"

using namespace hho;

const auto _pi = acos(-1.0);

// Test data-set for the poisson-dirichlet equation
struct test_data : public generic_test_data {
  scalar_type m, M;
  scalar_function_of_space concentration;
  scalar_function_of_space exact_viscosity;
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
    viscosity_construction constructor;
    function_reconstruction<degree> reconstructor(mesh);
    global_interpolator<degree> interpolator(reconstructor);
 
    // Interpolate the concentration
    auto CTF = interpolator.global_interpolant(data.concentration);
    auto c = reconstructor.reconstruct(CTF);
    
    // Construct the viscosity
    auto reconstruction = constructor.construct(c, data.m, data.M);
   
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
          auto exact_value_iqn = data.exact_viscosity(xQN);
          auto reconstructed_value_iqn = reconstruction(iT, xQN);
          auto diff = exact_value_iqn - reconstructed_value_iqn;

          error += wQN * (diff * diff);
        }
      }
    }
    
    error = std::pow(error, 0.5);
    
    // Error computation
    std::cout << " error = " << error << ". " << std::endl;
    
    return std::make_pair(error, std::move(CTF));
  }
};

int main() {

  // Create an instance of the tester and create tests
  tester T;
  
  // ----------------------------------------------------------------
  //                Viscosity construction test
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    
    // Oil viscosity
    data->m = 1.0;
    data->M = 41.0;
    
    // Exact concentration -- sin(pi x) sin(pi y)
    data->concentration = [](const point_type& x) {
      return sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    // Exact viscosity using the mixing rule
    data->exact_viscosity = [=](const point_type& x) {
      return data->m * std::pow(1.0 + (std::pow(data->M, 0.25) - 1.0) * data->concentration(x), -4.0);
    };
    
    T.add_test("Viscosity construction test", std::shared_ptr<generic_test_data>(data));
  }
  
  // Run the tests
  T.run_tests();
  
  return 0;
}

