// file: test_diffusion_neumann.cpp
// author: Daniel Anderson
//
// Tests for the HHO scheme for the diffusion equation with Neumann boundary conditions
//

#include "hho/common.hpp"
#include "hho/diffusion_neumann_solver.hpp"

#include "test_base.hpp"

using namespace hho;

const auto _pi = acos(-1.0);

// Test data-set for the diffusion-neumann equation
struct test_data : public generic_test_data {
  scalar_function_of_space exact_solution;
  cellwise_flux_function boundary_condition;
  cellwise_scalar_function source_term;
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
    const std::shared_ptr<test_data> dn_data = std::dynamic_pointer_cast<test_data>(g_data);
    const test_data& data = *dn_data;
    
    // Create solvers / tools
    diffusion_neumann_solver<degree> solver(mesh);
    global_interpolator<degree> interpolator(solver);
    integrator<degree> quad(solver);
    
    // Solve the scheme and compare with exact solution
    auto Xh = solver.solve(data.source_term, data.boundary_condition, data.kappa);
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
  //                     Mild anisotropy test
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- cos(pi x) cos(pi y)
    data->exact_solution = [](const point_type& x) { return cos(_pi * x.x()) * cos(_pi * x.y()); };
    
    // Diffusion tensor -- (1.5 0; 0 1.5)
    Eigen::Matrix<scalar_type, 2, 2> K;
    K << 1.5, 0.5, 0.5, 1.5;
    data->kappa = [=](const size_t iT, const point_type& pt) {
      return K;
    };
    
    // Neumann boundary value -- kappa grad . n_TF
    data->boundary_condition = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      auto kappa = data->kappa(0, x);
      vector_type grad { -_pi * cos(_pi * x.y()) * sin(_pi * x.x()), -_pi * cos(_pi * x.x()) * sin(_pi * x.y()) };
      return (kappa * grad).dot(nTF);
    };
    
    // Diffusive source term
    data->source_term = [](const size_t iT, const point_type& x) {
      return (_pi * _pi) * (3 * cos(_pi * x.x()) * cos(_pi * x.y()) - sin(_pi * x.x()) * sin(_pi * x.y()));        
    }; 
    
    T.add_test("Diffusion Neumann Mild Anisotropy", std::shared_ptr<generic_test_data>(data));
  }
  
  // ----------------------------------------------------------------
  //              Heterogeneous rotating anisotropy test
  // ----------------------------------------------------------------
  {
    const double offset = 2;    // Offset the domain to avoid the singularity in the tensor
  
    test_data* data = new test_data();
    // Exact solution -- sin(pi x) sin(pi y) - 4/pi^2
    data->exact_solution = [=](const point_type& x) {
      auto xx = x.x() + offset;
      auto yy = x.y() + offset;
      return sin(_pi * xx) * sin(_pi * yy) - 4.0 / (_pi * _pi);
    };
    
    // Diffusion tensor
    data->kappa = [=](const size_t iT, const point_type& x) {
      auto xx = x.x() + offset;
      auto yy = x.y() + offset;
      Eigen::Matrix<scalar_type, 2, 2> K;
      const auto r = 1.0 / (xx*xx + yy*yy);
      const auto a = 0.001 * xx*xx + yy*yy;
      const auto b = (0.001 - 1) * xx * yy;
      const auto c = xx*xx + 0.001 * yy*yy;
      K << r*a, r*b, r*b, r*c;
      return K;
    };
    
    // Neumann boundary value -- kappa grad . n_TF
    data->boundary_condition = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      auto xx = x.x() + offset;
      auto yy = x.y() + offset;
      auto kappa = data->kappa(0, x);
      vector_type grad {_pi * cos(_pi * xx) * sin(_pi * yy), _pi * cos(_pi * yy) * sin(_pi * xx) };
      return (kappa * grad).dot(nTF);
    };
    
    // Diffusive source term
    data->source_term = [=](const size_t iT, const point_type& pt) {
      const auto& _x = pt.to_vector();
      const auto x = _x(0) + offset, y = _x(1) + offset;
      return (999*std::pow(_pi,2)*x*y*cos(_pi*x)*cos(_pi*y))/(500.*(std::pow(x,2) + std::pow(y,2))) - 
       (999*_pi*std::pow(x,2)*y*cos(_pi*y)*sin(_pi*x))/(500.*std::pow(std::pow(x,2) + std::pow(y,2),2)) + 
       (2*_pi*y*(std::pow(x,2) + std::pow(y,2)/1000.)*cos(_pi*y)*sin(_pi*x))/std::pow(std::pow(x,2) + std::pow(y,2),2) + 
       (997*_pi*y*cos(_pi*y)*sin(_pi*x))/(1000.*(std::pow(x,2) + std::pow(y,2))) - 
       (999*_pi*x*std::pow(y,2)*cos(_pi*x)*sin(_pi*y))/(500.*std::pow(std::pow(x,2) + std::pow(y,2),2)) + 
       (2*_pi*x*(std::pow(x,2)/1000. + std::pow(y,2))*cos(_pi*x)*sin(_pi*y))/std::pow(std::pow(x,2) + std::pow(y,2),2) + 
       (997*_pi*x*cos(_pi*x)*sin(_pi*y))/(1000.*(std::pow(x,2) + std::pow(y,2))) + 
       (std::pow(_pi,2)*(std::pow(x,2) + std::pow(y,2)/1000.)*sin(_pi*x)*sin(_pi*y))/(std::pow(x,2) + std::pow(y,2)) + 
       (std::pow(_pi,2)*(std::pow(x,2)/1000. + std::pow(y,2))*sin(_pi*x)*sin(_pi*y))/(std::pow(x,2) + std::pow(y,2));       
    }; 
    
    T.add_test("Diffusion Neumann Heterogeneous Rotating Anisotropy", std::shared_ptr<generic_test_data>(data));
  }
  
  // Run the tests
  T.run_tests();
  
  return 0;
}
