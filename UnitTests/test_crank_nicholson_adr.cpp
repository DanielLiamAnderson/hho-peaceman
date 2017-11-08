// file: test_crank_nicholson_adr.cpp
// author: Daniel Anderson
//
// Tests for the HHO scheme for the time dependent
// Advection-Reaction-Diffusion equation using
// Crank Nicholson time stepping.
//

#include "hho/post_processing.hpp"
#include "hho/global_interpolator.hpp"
#include "hho/integrator.hpp"

#include "test_base.hpp"

#include "hho/peaceman/crank_nicholson_adr_solver.hpp"

#define SAVE_PLOTS 0

#define RANK (tensor_rank)0

using namespace hho;

const auto _pi = acos(-1.0);

const std::vector<int> num_steps = {16,32,64,128,256,512};

const std::vector<std::string> test_meshes = {
  "./meshes/mesh1_5.typ1",
  "./meshes/mesh2_5.typ1",
  "./meshes/mesh4_1_5.typ1",
  "./meshes/pi6_tiltedhexagonal_5.typ1",
};

struct time_dependent_adr_data {
  scalar_type t0;
  scalar_type t1;
  scalar_function_of_spacetime exact_solution;
  cellwise_scalar_function_of_time source_term;
  cellwise_flux_function_of_time boundary; 
  cellwise_tensor_function_of_time kappa;
  cellwise_vector_function_of_time beta;
  cellwise_flux_function_of_time flux;
  cellwise_scalar_function_of_time mu;
};

// Run a test using the given data at the given degree
template<size_t spatial_degree>
std::pair<std::vector<scalar_type>, std::vector<scalar_type>> test_degree(const time_dependent_adr_data& data,
    const std::shared_ptr<mesh_type>& mesh, const std::string& mesh_filename, const std::string& test_name) {

  // Print header
  std::cout << std::endl << cyan << bold << "\t*** Testing degree " << spatial_degree << " ***" << reset << std::endl;

  // Create solvers and tools
  crank_nicholson_adr_solver<spatial_degree> solver(mesh);
  global_interpolator<spatial_degree> interpolator(mesh);
  integrator<spatial_degree> quad(interpolator);
  
  // Store results
  std::pair<std::vector<scalar_type>, std::vector<scalar_type>> results;
  
  // Interpolate the exact solution
  auto exact_final_solution = [=](const point_type& x) {
    return data.exact_solution(data.t1, x);
  };
  auto XTF = interpolator.global_interpolant(exact_final_solution);
  
  // Generate initial condition
  cellwise_scalar_function initial_condition = [=](const size_t iT, const point_type& x) {
    return data.exact_solution(data.t0, x);
  };
  
  // Solve for each value of num_steps
  for (size_t j = 0; j < num_steps.size(); j++) {
    const auto n_steps = num_steps[j];
    
    // Print
    auto delta_t = (data.t1 - data.t0) / n_steps;
    std::cout << "\t\tdelta_t = " << std::fixed << std::setprecision(3) << delta_t << "... " << std::flush;

    // Solve the system
    auto Xh = solver.solve(data.t0, data.t1, n_steps, data.source_term, data.boundary, data.kappa, data.beta,
    data.flux, data.mu, initial_condition, false);
    
    // Compare with the exact solution
    auto diff = Xh.back() - XTF;
    auto error = quad.l2_norm(diff);
    
    std::cout << "error = " << std::scientific << error << std::endl;
    
    // Store results
    results.first.push_back(delta_t);
    results.second.push_back(error);
    
    // Save the output
#if SAVE_PLOTS
    std::cout << green << "\t\tSaving plots..." << reset << std::endl;
    for (size_t i = 0; i < Xh.size(); i++) {
      auto mesh_name = boost::filesystem::path(mesh_filename).stem().string();
      std::string filename = "./output/time_dependent_tests/crank-nicholson/" + test_name + "-"
      + mesh_name + "-degree" + std::to_string(spatial_degree) + "-" + std::to_string(i) + ".vtu";
      std::replace(filename.begin(), filename.end(), ' ', '_');
      post_processing<spatial_degree, RANK>(filename, solver.get_mesh(), Xh[i]);
    }
#endif  // SAVE_PLOTS
  }
  
  return results;
}

// Run a test using the given data
void test(const time_dependent_adr_data& data, const std::string test_name) {
  
  // Print test header
  std::cout << green << bold << std::string(80, '-') << std::endl;
  std::cout << "\t Running test data set -- " << test_name << std::endl;
  std::cout << std::string(80, '-') << std::endl << reset;

  for (const auto& mesh_filename : test_meshes) {
  
    // Print header
    std::cout << std::endl << red << bold << "****** Testing Mesh " << mesh_filename << " ******" << reset << std::endl;
  
    auto mesh = std::make_shared<mesh_type>();
    load_mesh_from_file(mesh_filename, mesh);
    
    std::vector<scalar_type> step_sizes(num_steps.size());
    std::vector<std::vector<scalar_type>> errors(5, std::vector<scalar_type>(num_steps.size()));
    
    auto res0 = test_degree<0>(data, mesh, mesh_filename, test_name);
    auto res1 = test_degree<1>(data, mesh, mesh_filename, test_name);
    auto res2 = test_degree<2>(data, mesh, mesh_filename, test_name);
    auto res3 = test_degree<3>(data, mesh, mesh_filename, test_name);
    auto res4 = test_degree<4>(data, mesh, mesh_filename, test_name);
    
    step_sizes = res0.first;
    errors[0] = res0.second;
    errors[1] = res1.second;
    errors[2] = res2.second;
    errors[3] = res3.second;
    errors[4] = res4.second;
    
    // Save the results
    std::cout << green << "Saving convergence data for " << mesh_filename << reset << std::endl;
    
    // Output filenames
    auto mesh_name = boost::filesystem::path(mesh_filename).stem().string();
    std::string outfilename = test_name + "-" + mesh_name;
    std::replace(outfilename.begin(), outfilename.end(), ' ', '_');
    
    // Create directory if it doesn't exist
    auto datafilename = data_directory + outfilename;
    auto path = boost::filesystem::path(datafilename);
    if (!boost::filesystem::exists(path.parent_path())) {
      boost::filesystem::create_directory(path.parent_path());
    }

    // Write the file
    std::ofstream outfile(datafilename);
    
    outfile << mesh_filename << std::endl;
    outfile << "$\\Delta t$" << std::endl;
    outfile << "$L^2$ Error" << std::endl;
    for (size_t i = 0; i < step_sizes.size(); i++) {
      outfile << step_sizes[i] << ' ';
    }
    outfile << std::endl;
    
    for (size_t k = 0; k <= 4; k++) {
      outfile << "$k = " << k << "$" << std::endl;
      for (size_t i = 0; i < step_sizes.size(); i++) {
        outfile << errors[k][i] << ' ';
      }
      outfile << std::endl;
    }
    outfile.close();
  
  }
}

int main() {
  
  //---------------------------------------------------------------------------
  //              Simple diffusion -- Advection free test
  {
    time_dependent_adr_data data;
    
    // Parameters
    data.t0 = 0;
    data.t1 = 2.5;
    
    // Exact solution -- t cos(pi x) cos(pi y)
    data.exact_solution = [](const scalar_type t, const point_type& x) {
      return sin(_pi * t) * sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    // Neumann boundary condition
    data.boundary = [](const scalar_type t) {
      return [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
        vector_type grad {sin(_pi * t) * _pi * cos(_pi * x.x()) * sin(_pi * x.y()),
                          sin(_pi * t) * _pi * cos(_pi * x.y()) * sin(_pi * x.x()) };
        return grad.dot(nTF);
      };
    };
    
    // Diffusion tensor -- Identity
    data.kappa = [](const scalar_type t) {
      return [=](const size_t iT, const point_type& x) {
        return Eigen::Matrix<scalar_type, 2, 2>::Identity();
      };
    };
    
    // Velocity -- none
    data.beta = [](const scalar_type t) {
      return [=](const size_t iT, const point_type& x) {
        return vector_type {0.0, 0.0};
      };
    };
    
    // Flux
    data.flux = [=](const scalar_type t) {
      return [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
        return 0.0;
      };
    };
    
    // Reaction -- none
    data.mu = [](const scalar_type t) {
      return [=](const size_t iT, const point_type& x) {
        return 0.0;
      };
    };
    
    // Source term
    data.source_term = [=](const scalar_type t) {
      return [=](const size_t iT, const point_type& x) {
        return sin(_pi * x.x()) * sin(_pi * x.y()) * (_pi * cos(_pi * t) + 2 * _pi * _pi * sin(_pi * t));
      };
    };
  
    test(data, "Crank Nicholson diffusion");
  }
  
  //---------------------------------------------------------------------------
  //              Advection test
  {
    time_dependent_adr_data data;
    
    // Parameters
    data.t0 = 0;
    data.t1 = 2.5;
    
    // Exact solution -- t cos(pi x) cos(pi y)
    data.exact_solution = [](const scalar_type t, const point_type& x) {
      return sin(_pi * t) * sin(_pi * x.x()) * sin(_pi * x.y());
    };
    
    // Exact gradient
    auto grad_u = [](const scalar_type t, const point_type& x) {
      vector_type grad {sin(_pi * t) * _pi * cos(_pi * x.x()) * sin(_pi * x.y()),
                        sin(_pi * t) * _pi * cos(_pi * x.y()) * sin(_pi * x.x()) };
      return grad;
    };
    
    // Diffusion tensor -- 10^-3
    data.kappa = [](const scalar_type t) {
      return [=](const size_t iT, const point_type& x) {
        return 1e-3 * Eigen::Matrix<scalar_type, 2, 2>::Identity();
      };
    };
    
    // Velocity -- (x(1-x), y(1-y))
    data.beta = [](const scalar_type t) {
      return [=](const size_t iT, const point_type& x) {
        return vector_type {x.x() * (1.0 - x.x()), x.y() * (1.0 - x.y())};
      };
    };
    
    // Flux
    data.flux = [=](const scalar_type t) {
      auto beta_t = data.beta(t);
      return [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
        return beta_t(iT, x).dot(nTF);
      };
    };
    
    // Divergence of beta
    auto div_beta = [](const scalar_type t, const point_type& x) {
      return 2 - 2 * x.x() - 2 * x.y();
    };
    
    // Neumann boundary condition
    data.boundary = [=](const scalar_type t) {
      auto K_T = data.kappa(t);
      auto beta_T = data.beta(t);
      return [=](const size_t iT, const size_t iF_loc, const point_type& x, const vector_type& nTF) {
        auto u = data.exact_solution(t, x);
        auto grad = grad_u(t, x);
        auto K = K_T(0, x);
        auto beta = beta_T(0, x);
        return (K * grad).dot(nTF) - (beta.dot(nTF) * u);
      };
    };
    
    // Reaction -- uniform
    data.mu = [](const scalar_type t) {
      return [=](const size_t iT, const point_type& x) {
        return 1.0;
      };
    };
    
    // Source term
    data.source_term = [=](const scalar_type t) {
      auto K_T = data.kappa(t);
      auto beta_T = data.beta(t);
      auto mu_T = data.mu(t);
      return [=](const size_t iT, const point_type& x) {
        auto u = data.exact_solution(t, x);
        auto grad = grad_u(t, x);
        auto beta = beta_T(0, x);
        auto div = div_beta(t, x);
        auto mu = mu_T(0, x);
        return _pi * cos(_pi * t) * sin(_pi * x.x()) * sin(_pi * x.y())         // u_t
              + 1e-3 * 2 * _pi * _pi * u                                        // - div(K grad[u])
              + div * u + beta.dot(grad)                                        // div(beta u)
              + mu * u;                                                         // mu u
      };
    };
  
    test(data, "Crank Nicholson Advection");
  }
  
  return 0;
}


