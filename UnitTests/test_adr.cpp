// file: test_adr.cpp
// author: Daniel Anderson
//
// Tests for the HHO scheme for the Advection-Reaction-Diffusion equation
//

#include "hho/adr_solver.hpp"
#include "hho/common.hpp"

#include "test_base.hpp"

using namespace hho;

const auto _pi = acos(-1.0);

// Test data-set for the advection-diffusion-reaction equation
struct test_data : public generic_test_data {
  scalar_function_of_space exact_solution;
  cellwise_flux_function boundary_condition;
  cellwise_scalar_function source_term;
  cellwise_tensor_function kappa;
  cellwise_scalar_function mu;
  cellwise_vector_function beta;
  cellwise_flux_function flux;
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
    adr_solver<degree> solver(mesh);
    global_interpolator<degree> interpolator(solver);
    integrator<degree> quad(solver);
    
    // Solve the scheme and compare with exact solution
    auto Xh = solver.solve(data.source_term, data.boundary_condition, data.kappa, data.beta, data.flux, data.mu);
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
  //                Heterogeneous diffusion and reaction
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- sin(pi x) sin(pi y)
    data->exact_solution = [](const point_type& x) {
      return sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    // Diffusion tensor -- Identity
    data->kappa = [](const size_t iT, const point_type& pt) {
      return Eigen::Matrix<scalar_type, 2, 2>::Identity();
    };
    
    // Neumann boundary value -- kappa grad . n_TF
    data->boundary_condition = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      auto kappa = data->kappa(0, x);
      vector_type grad {_pi * cos(_pi * x.x()) * sin(_pi * x.y()), _pi * cos(_pi * x.y()) * sin(_pi * x.x()) };
      return (kappa * grad).dot(nTF);
    };
    
    // Reaction terms -- mu = 1
    data->mu = [](const size_t iT, const point_type& x) {
      return 1.0;
    };
    
    // Diffusive source term
    data->source_term = [](const size_t iT, const point_type& x) {
      return _pi * ( cos(_pi * x.x()) * sin(_pi * x.y()) * (0.5 - x.y())
                    + (x.x() - 0.5) * sin(_pi * x.x()) * cos(_pi * x.y()) )
                    + 1.0 * sin(_pi * x.x()) * sin(_pi * x.y())
                    + 2.0 * (_pi * _pi) * sin(_pi * x.x()) * sin(_pi * x.y());          
    }; 

    // Advective velocity
    data->beta = [](const size_t iT, const point_type& x) {
      return vector_type{0.5 - x.y(), x.x() - 0.5};
    };
    
    // Flux of the advective velocity
    data->flux = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      return data->beta(iT, x).dot(nTF);
    };
    
    T.add_test("ADR isotropic diffusion", std::shared_ptr<generic_test_data>(data));
  }
  
  // ----------------------------------------------------------------
  //                Minor diffusion -- advection dominated
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- sin(pi x) sin(pi y)
    data->exact_solution = [](const point_type& x) {
      return sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    // Diffusion tensor -- Identity
    data->kappa = [](const size_t iT, const point_type& pt) {
      return 1e-3 * Eigen::Matrix<scalar_type, 2, 2>::Identity();
    };
    
    // Neumann boundary value -- kappa grad . n_TF
    data->boundary_condition = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      auto kappa = data->kappa(0, x);
      vector_type grad {_pi * cos(_pi * x.x()) * sin(_pi * x.y()), _pi * cos(_pi * x.y()) * sin(_pi * x.x()) };
      return (kappa * grad).dot(nTF);
    };
    
    // Reaction terms -- mu = 1
    data->mu = [](const size_t iT, const point_type& x) {
      return 1.0;
    };
    
    // Diffusive source term
    data->source_term = [](const size_t iT, const point_type& x) {
      return _pi * ( cos(_pi * x.x()) * sin(_pi * x.y()) * (0.5 - x.y())
                    + (x.x() - 0.5) * sin(_pi * x.x()) * cos(_pi * x.y()) )
                    + 1.0 * sin(_pi * x.x()) * sin(_pi * x.y())
                    + 2.0 * 1e-3 * (_pi * _pi) * sin(_pi * x.x()) * sin(_pi * x.y());          
    }; 

    // Advective velocity
    data->beta = [](const size_t iT, const point_type& x) {
      return vector_type{0.5 - x.y(), x.x() - 0.5};
    };
    
    // Flux of the advective velocity
    data->flux = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      return data->beta(iT, x).dot(nTF);
    };
    
    T.add_test("ADR advection dominated", std::shared_ptr<generic_test_data>(data));
  }

  // ----------------------------------------------------------------
  //                Nonzero boundary advection
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- cos(pi x) cos(pi y)
    data->exact_solution = [](const point_type& x) {
      return cos(_pi * x.x()) * cos(_pi * x.y());
    };
    
    // Gradient of the exact solution
    auto grad_u = [](const point_type& x) {
      vector_type grad { - _pi * cos(_pi * x.y()) * sin(_pi * x.x()), - _pi * cos(_pi * x.x()) * sin(_pi * x.y()) };
      return grad;
    };
    
    // Diffusion tensor -- Identity
    data->kappa = [](const size_t iT, const point_type& pt) {
      return Eigen::Matrix<scalar_type, 2, 2>::Identity();
    };
    
    // Advective velocity
    data->beta = [](const size_t iT, const point_type& x) {
      return vector_type{0.5 - x.y(), x.x() - 0.5};
    };
    
    // Flux of the advective velocity
    data->flux = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      return data->beta(iT, x).dot(nTF);
    };
    
    // Neumann boundary value -- kappa grad . n_TF - (beta . n) u
    data->boundary_condition = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      auto kappa = data->kappa(0, x);
      auto beta = data->beta(0, x);
      auto u = data->exact_solution(x);
      auto grad = grad_u(x);
      return (kappa * grad).dot(nTF) - (beta.dot(nTF)) * u;
    };
    
    // Reaction terms -- mu = 1
    data->mu = [](const size_t iT, const point_type& x) {
      return 1.0;
    };
    
    // Diffusive source term
    data->source_term = [=](const size_t iT, const point_type& x) {
      auto u = data->exact_solution(x);
      auto beta = data->beta(0, x);
      auto mu = data->mu(0, x);
      auto grad = grad_u(x);
                        
      return 2.0 * (_pi*_pi) * u + beta.dot(grad) + mu * u;       
    }; 
    
    T.add_test("ADR nonzero boundary", std::shared_ptr<generic_test_data>(data));
  }

  // ----------------------------------------------------------------
  //                Non divergence-free advection
  // ----------------------------------------------------------------
  {
    test_data* data = new test_data();
    // Exact solution -- sin(pi x) sin(pi y)
    data->exact_solution = [](const point_type& x) {
      return sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    // Gradient of the exact solution
    auto grad_u = [](const point_type& x) {
      vector_type grad {_pi * cos(_pi * x.x()) * sin(_pi * x.y()), _pi * cos(_pi * x.y()) * sin(_pi * x.x()) };
      return grad;
    };
    
    // Divergence of the velocity
    auto div_beta = [](const point_type& x) {
      return 2.0 - 2 * x.x() - 2 * x.y();
    };
    
    // Diffusion tensor -- Identity
    data->kappa = [](const size_t iT, const point_type& pt) {
      return Eigen::Matrix<scalar_type, 2, 2>::Identity();
    };
    
    // Advective velocity
    data->beta = [](const size_t iT, const point_type& x) {
      return vector_type{x.x() * (1 - x.x()), x.y() * (1 - x.y())};
    };
    
    // Flux of the advective velocity
    data->flux = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      return data->beta(iT, x).dot(nTF);
    };
    
    // Neumann boundary value -- kappa grad . n_TF - (beta . n) u
    data->boundary_condition = [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
      auto kappa = data->kappa(0, x);
      auto grad = grad_u(x);
      return (kappa * grad).dot(nTF);
    };
    
    // Reaction terms -- mu = 1
    data->mu = [](const size_t iT, const point_type& x) {
      return 1.0;
    };
    
    // Diffusive source term
    data->source_term = [=](const size_t iT, const point_type& x) {
      auto div = div_beta(x);
      auto u = data->exact_solution(x);
      auto beta = data->beta(0, x);
      auto mu = data->mu(0, x);
      auto grad = grad_u(x);
                        
      return 2.0 * (_pi*_pi) * u + div * u + beta.dot(grad) + mu * u;
    }; 
    
    T.add_test("ADR non divergence free", std::shared_ptr<generic_test_data>(data));
  }
  
  // Run the tests
  T.run_tests();
  
  return 0;
}


