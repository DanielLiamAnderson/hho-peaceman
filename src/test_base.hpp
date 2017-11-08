// file: test_base.hpp
// author: Daniel Anderson
//
// Common elements required for convergence testing.
// 
// To perform convergence testing, one should inherit from this class and override
// the test0..3 methods. Testing can then be performed by providing the test
// class with a list of data sets, comprising of descendents of the generic_test_data
// class.
//
// See test_adr.cpp for an example.
//

#pragma once

#include <iostream>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include <boost/timer.hpp>

#include "hho/common.hpp"

#include "hho/global_interpolator.hpp"
#include "hho/integrator.hpp"
#include "hho/post_processing.hpp"

// http://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal
// Provides an easy interface for setting the terminal colours.
namespace Color {
    enum Code {
        BOLD        = 1,
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        FG_CYAN     = 36,
        FG_YELLOW   = 93,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49,
        RESET       = 0,
    };
    class Modifier {
        Code code;
    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream&
        operator<<(std::ostream& os, const Modifier& mod) {
            return os << "\033[" << mod.code << "m";
        }
    };
}

namespace hho {

const std::string data_directory = "./output/convergence/";
const std::string plot_directory = "./output/plots/";

constexpr int max_degree = 3;

// Terminal colours
Color::Modifier green(Color::FG_GREEN);
Color::Modifier def(Color::FG_DEFAULT);
Color::Modifier bold(Color::BOLD);
Color::Modifier reset(Color::RESET);
Color::Modifier cyan(Color::FG_CYAN);
Color::Modifier yellow(Color::FG_YELLOW);
Color::Modifier red(Color::FG_RED);

// Loads a mesh from the given file
//
// Arguments:
//  mesh_filename  the filename of the mesh to load
//  mesh           a pointer to the mesh data structure to store the loaded mesh in
void load_mesh_from_file(const std::string& mesh_filename, std::shared_ptr<mesh_type> mesh) {
  assert(::boost::filesystem::exists(mesh_filename));
  fvca5_mesh_loader<double, 2> loader;
  loader.read_mesh(mesh_filename);
  loader.populate_mesh(*mesh);
}

// Generic superclass for test data sets
struct generic_test_data {
  virtual ~generic_test_data() { }
};

// Pointers for test data
typedef const std::shared_ptr<mesh_type> mesh_ptr;
typedef const std::shared_ptr<generic_test_data> data_ptr;
typedef std::pair<double, dynamic_vector<scalar_type>> test_result;

// Hacky define to override tests in subclass
#define OVERRIDE_TESTS \
  test_result test0(mesh_ptr& mesh, data_ptr& data) { return test<0>(mesh, data); };  \
  test_result test1(mesh_ptr& mesh, data_ptr& data) { return test<1>(mesh, data); };  \
  test_result test2(mesh_ptr& mesh, data_ptr& data) { return test<2>(mesh, data); };  \
  test_result test3(mesh_ptr& mesh, data_ptr& data) { return test<3>(mesh, data); };  \

// Generic testing class -- Should be subclassed to implement tests for specific schemes
class generic_tester {
 public:
  // Runs all of the tests
  void run_tests() {
    boost::timer time_me;
    for (const auto& test : tests) {
      run_test(test);
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << yellow << bold << "Total time elapsed: " << time_me.elapsed() << "sec" << reset << std::endl;
  }
 
  // Add a test to the test-set
  void add_test(const std::string& name, const std::shared_ptr<generic_test_data>& data) {
    tests.emplace_back(name, data);
  }
  
  // Set the meshes to be used
  void set_meshes(std::vector<std::vector<std::string>> mesh_set,
    std::vector<std::string> names) {
    test_mesh_sets = std::move(mesh_set);
    mesh_names = std::move(names);
  }
 
 protected:
  // Test units for each degree -- must override seperately
  virtual test_result test0(const std::shared_ptr<mesh_type>& mesh, const std::shared_ptr<generic_test_data>& data) = 0;
  virtual test_result test1(const std::shared_ptr<mesh_type>& mesh, const std::shared_ptr<generic_test_data>& data) = 0;
  virtual test_result test2(const std::shared_ptr<mesh_type>& mesh, const std::shared_ptr<generic_test_data>& data) = 0;
  virtual test_result test3(const std::shared_ptr<mesh_type>& mesh, const std::shared_ptr<generic_test_data>& data) = 0;
 
 private:
  // Run the given test
  void run_test(const std::pair<std::string, std::shared_ptr<generic_test_data>>& test) {
  
    const auto& name = test.first;
    const auto& data = test.second;
    
    // Print test header
    std::cout << green << bold << std::string(80, '-') << std::endl;
    std::cout << "\t\t Running test data set -- " << name << std::endl;
    std::cout << std::string(80, '-') << std::endl << reset;
  
    std::cout << std::scientific << std::setprecision(5);

    // Test each mesh set
    for (size_t j = 0; j < test_mesh_sets.size(); j++) {
    
      std::cout << std::endl << cyan << bold << "*** Testing mesh set " << mesh_names[j] << " ***" << reset << std::endl;
      const auto& mesh_set = test_mesh_sets[j];
      
      std::vector<scalar_type> mesh_sizes(mesh_set.size());
      std::vector<std::vector<scalar_type>> errors(5, std::vector<scalar_type>(mesh_set.size()));
   
      // Output filenames
      std::string outfilename = name + "-" + mesh_names[j];
      std::replace(outfilename.begin(), outfilename.end(), ' ', '_');
   
      // Test each mesh in this set
      for (size_t i = 0; i < mesh_set.size(); i++) {
      
        const auto& filename = mesh_set[i];
      
        // Load the mesh
        auto mesh = std::make_shared<mesh_type>();
        load_mesh_from_file(filename, mesh);
        mesh_sizes[i] = mesh_h(*mesh);
      
        std::cout << "\tTesting " << filename << ", h = " << mesh_sizes[i] << std::endl;
        
        // Test degree k = 0, 1, 2, 3
        auto result0 = test0(mesh, data);
        auto result1 = test1(mesh, data);
        auto result2 = test2(mesh, data);
        auto result3 = test3(mesh, data);
        
        // Save the error
        errors[0][i] = result0.first;
        errors[1][i] = result1.first;
        errors[2][i] = result2.first;
        errors[3][i] = result3.first;
        
        // Saving plots takes a long time and a lot of space. Define SAVE_SOLUTION_PLOTS
        // if you really want to enable this.
#ifdef SAVE_SOLUTION_PLOTS
        {
          std::cout << green << "\t\tSaving plots..." << reset << std::endl;
          assert_directory(plot_directory);
          auto mesh_filename = ::boost::filesystem::path(filename).stem().string();
          auto plotfilename = plot_directory + outfilename + "-" + mesh_filename;
          post_processing<0>(plotfilename + "-degree0.vtu", *mesh, result0.second);
          post_processing<1>(plotfilename + "-degree1.vtu", *mesh, result1.second);
          post_processing<2>(plotfilename + "-degree2.vtu", *mesh, result2.second);
          post_processing<3>(plotfilename + "-degree3.vtu", *mesh, result3.second);
        }
#endif
      }

      // Save the results
      std::cout << green << "Saving convergence data for " << mesh_names[j] << reset << std::endl;
      
      // Create directory if it doesn't exist
      assert_directory(data_directory);
      auto datafilename = data_directory + outfilename;
      
      // Write the file
      std::ofstream outfile(datafilename);
      
      outfile << mesh_names[j] << std::endl;
      outfile << "$h$" << std::endl;
      outfile << "$L^2$ Error" << std::endl;
      for (size_t i = 0; i < mesh_set.size(); i++) {
        outfile << mesh_sizes[i] << ' ';
      }
      outfile << std::endl;
      
      for (size_t k = 0; k <= max_degree; k++) {
        outfile << "$k = " << k << "$" << std::endl;
        for (size_t i = 0; i < mesh_set.size(); i++) {
          outfile << errors[k][i] << ' ';
        }
        outfile << std::endl;
      }
      outfile.close();
    }
  }
 
  // A list of tests to run
  std::vector<std::pair<std::string, std::shared_ptr<generic_test_data>>> tests;
  
  // A list of meshes
  // Standard 1.0 * 1.0 domain meshes
  std::vector<std::vector<std::string>> test_mesh_sets = {
    // Triangular mesh
    {"./meshes/mesh1_1.typ1",
     "./meshes/mesh1_2.typ1",
     "./meshes/mesh1_3.typ1",
     "./meshes/mesh1_4.typ1",
     "./meshes/mesh1_5.typ1"},
    // Square mesh
    {"./meshes/mesh2_1.typ1",
     "./meshes/mesh2_2.typ1",
     "./meshes/mesh2_3.typ1",
     "./meshes/mesh2_4.typ1",
     "./meshes/mesh2_5.typ1"},
    // Kershaw meshes
    {"./meshes/mesh4_1_1.typ1",
     "./meshes/mesh4_1_2.typ1",
     "./meshes/mesh4_1_3.typ1",
     "./meshes/mesh4_1_4.typ1",
     "./meshes/mesh4_1_5.typ1"},
    // Skewed hexagonal mesh
    {"./meshes/pi6_tiltedhexagonal_1.typ1",
     "./meshes/pi6_tiltedhexagonal_2.typ1",
     "./meshes/pi6_tiltedhexagonal_3.typ1",
     "./meshes/pi6_tiltedhexagonal_4.typ1",
     "./meshes/pi6_tiltedhexagonal_5.typ1",}
  };

  // Titles of mesh varieties to display in output
  std::vector<std::string> mesh_names = {
    "Triangular meshes",
    "Square meshes",
    "Kershaw meshes",
    "Tilted hexagonal meshes"
  };
};


} // namespace hho
