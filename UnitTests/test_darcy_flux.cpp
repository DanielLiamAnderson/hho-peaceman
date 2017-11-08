// file: test_darcy_flux.cpp
// author: Daniel Anderson
//
// Tests the flux reconstruction of the Darcy velocity
//

#include "hho/common.hpp"
#include "hho/peaceman/darcy_velocity_reconstruction.hpp"

#include "test_base.hpp"

using namespace hho;

const auto _pi = acos(-1.0);

// Test data-set for the poisson-dirichlet equation
struct test_data : public generic_test_data {
  scalar_function_of_space exact_solution;
  cellwise_vector_function exact_gradient;
  cellwise_tensor_function kappa;
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
    const std::shared_ptr<test_data> flux_data = std::dynamic_pointer_cast<test_data>(g_data);
    const test_data& data = *flux_data;
    
    // Create solvers / tools
    darcy_velocity_reconstruction<degree> reconstructor(mesh);
    global_interpolator<degree> interpolator(reconstructor);
 
    // Interpolate the exact solution
    auto XTF = interpolator.global_interpolant(data.exact_solution);
    
    // Reconstruct the fluxes
    auto fluxes = reconstructor.reconstruct(XTF, data.kappa).second;
   
    // Compute the error
    const auto& Th = *mesh;
    auto error = 0.0;
    
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = std::distance(Th.cells_begin(), it_T);
      const auto& E = reconstructor.get_local_elements()[iT];
      auto xT = barycenter(Th, T);
      auto FT = faces(Th, T); 

      // Compute the error on this face
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        const auto& F = *it_F;
        const auto iF_loc = std::distance(FT.begin(), it_F);
        const auto& qr_F = E.quadrature_rule_on_face(iF_loc);
        
        auto error_F = 0.0;
        
        for (size_t iQN = 0; iQN < qr_F.first.size(); iQN++) {
          const auto& xQN = qr_F.first[iQN];
          const auto& wQN = qr_F.second[iQN];
          
          auto nTF = normal(Th, F, xT);
          auto kappa_T_iQN = data.kappa(iT, xQN);
          auto grad_iQN = data.exact_gradient(iT, xQN);
          
          // Compute exact flux at xQN
          auto exact_flux_iQN = - (kappa_T_iQN * grad_iQN).dot(nTF);
          
          // Compute reconstructed flux at xQN
          auto reconstructed_flux_iQN = fluxes(iT, iF_loc, xQN, nTF);

          // Accumulate
          auto diff = exact_flux_iQN - reconstructed_flux_iQN;
          error_F += wQN * (diff * diff);
        }
        
        // Update the maximum error
        error_F = std::pow(error_F, 0.5);
        error = std::max(error, error_F);
        //error += error_F;
      }
    }
    
    // Error computation
    std::cout << " error = " << error << ". " << std::endl;
    
    return std::make_pair(error, std::move(XTF));
  }
};

int main() {

  // Create an instance of the tester and create tests
  tester T;
  
  // ----------------------------------------------------------------
  //           Isotropic diffusion fluxes on sin sin
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- sin(pi x) sin(pi y)
    data->exact_solution = [](const point_type& x) {
      return sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    data->exact_gradient = [](const size_t iT, const point_type& x) {
      vector_type grad {_pi * cos(_pi * x.x()) * sin(_pi * x.y()), _pi * cos(_pi * x.y()) * sin(_pi * x.x()) };
      return grad;
    };
    
    data->kappa = [](const size_t iT, const point_type& x) {
      return Eigen::Matrix<scalar_type, 2, 2>::Identity();
    };
    
    T.add_test("Isotropic diffusion fluxes", std::shared_ptr<generic_test_data>(data));
  }
  
  // Run the tests
  T.run_tests();
  
  return 0;
}

