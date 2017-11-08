// file: test_diffusion_dispersion_tensor.cpp
// author: Daniel Anderson
//
// Tests the diffusion-dispersion tensor construction.
//

#include "hho/common.hpp"
#include "hho/global_interpolator.hpp"
#include "hho/peaceman/darcy_velocity_reconstruction.hpp"
#include "hho/peaceman/diffusion_dispersion_tensor_construction.hpp"

#include "test_base.hpp"

using namespace hho;

const auto _pi = acos(-1.0);

// Test data-set for the advection-diffusion-reaction equation
struct test_data : public generic_test_data {
  scalar_function_of_space F;   // The potential function from which to reconstruct a velocity / fluxes
  scalar_type dm, dl, dt;
  cellwise_scalar_function phi;
  cellwise_tensor_function kappa;
  cellwise_tensor_function exact_tensor;
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
    const std::shared_ptr<test_data> adr_data = std::dynamic_pointer_cast<test_data>(g_data);
    const test_data& data = *adr_data;
    
    // Create solvers / tools
    global_interpolator<degree> interpolator(mesh);
    integrator<degree> quad(interpolator);
    darcy_velocity_reconstruction<degree> darcy_reconstructor(interpolator);
    diffusion_dispersion_tensor_construction constructor;
    
    // Reconstruct the velocity and the fluxes to use
    auto Fh = interpolator.global_interpolant(data.F);
    auto darcy_velocity = darcy_reconstructor.reconstruct(Fh, data.kappa);
    
    // Use the reconstructed velocity and fluxes
    auto& velocity = darcy_velocity.first;
    
    // Construct the diffusion dispersion tensor
    auto D_approx = constructor.construct(velocity, data.phi, data.dm, data.dt, data.dl);
    
    // Compute the error
    const auto& Th = *mesh;
    auto error = 0.0;
    
    for (auto it_T = Th.cells_begin(); it_T != Th.cells_end(); it_T++) {
      const auto& T = *it_T;
      const auto iT = std::distance(Th.cells_begin(), it_T);
      const auto& E = darcy_reconstructor.get_local_elements()[iT];
      auto FT = faces(Th, T); 

      // Compute the error in this subelement
      for (auto it_F = FT.begin(); it_F != FT.end(); it_F++) {
        const auto iF_loc = std::distance(FT.begin(), it_F);
        const auto& qr_PTF = E.quadrature_rule_on_pyramid(iF_loc);

        for (size_t iQN = 0; iQN < qr_PTF.first.size(); iQN++) {
          const auto& xQN = qr_PTF.first[iQN];
          const auto& wQN = qr_PTF.second[iQN];
          
          // Compute the exact value and reconstructed value
          auto approx_value_iqn = D_approx(iT, xQN);
          auto exact_value_iqn = data.exact_tensor(0, xQN);
          auto err = (approx_value_iqn - exact_value_iqn);
          auto diff = err.template lpNorm<2>();

          error += wQN * (diff * diff);
        }
      }
    }
    
    error = std::pow(error, 0.5);
    std::cout << " error = " << error << ". " << std::endl;
    
    return std::make_pair(error, std::move(Fh));
  }
};

int main() {

  // Create an instance of the tester and create tests
  tester T;
  
  // ----------------------------------------------------------------
  //               Diffusion dispersion tensor test
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    
    // Reconstruct a velocity to use
    // F = - x^2/2 + x^3/3 - y^2/2 + y^3/3
    // Gives rise to beta = (x(1-x), y(1-y))
    data->F = [](const point_type& x) {
      return sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    // The diffusion coefficients
    data->dm = 10.0;
    data->dt = 50.0;
    data->dl = 5.0;
    
    // Permability of the medium
    data->phi = [](const size_t iT, const point_type& x) { 
      return 0.1;
    };
    
    // Constant permeability
    data->kappa = [](const size_t iT, const point_type& x) {
      return Eigen::Matrix<scalar_type, 2, 2>::Identity();
    };
    
    // Exact value of the diffusion-dispersion tensor
    auto eye = Eigen::Matrix<scalar_type, 2, 2>::Identity();
    data->exact_tensor = [=](const size_t iT, const point_type& x) -> Eigen::Matrix<scalar_type, 2, 2> {
      auto beta = vector_type{- _pi * cos(_pi * x.x())*sin(_pi * x.y()), - _pi * sin(_pi * x.x()) * cos(_pi * x.y())};
      if (beta.norm() < 1e-8) {
        return data->phi(0, x) * data->dm * eye;
      } else {
        auto E = (beta * beta.transpose()) / beta.norm();
        return data->phi(0, x) * (data->dm * eye + data->dl * E + data->dt * (eye - E));
      }
    };
    
    T.add_test("Diffusion dispersion tensor test", std::shared_ptr<generic_test_data>(data));
  }
  
  // Run the tests
  T.run_tests();
  
  return 0;
}


