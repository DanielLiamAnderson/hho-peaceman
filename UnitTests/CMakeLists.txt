# -------------------------------------------------------------------------------
#                   CMAKE TARGETS FOR HHO PEACEMAN UNIT TESTS
# -------------------------------------------------------------------------------

# List of unit tests to build
SET(HHO_UNIT_TESTS
  test_diffusion_neumann;
  test_adr;
  test_adr_on_reconstructed_velocity;
  test_crank_nicholson_adr;
  test_darcy_velocity;
  test_darcy_flux;
  test_diffusion_dispersion_tensor;
  test_function_reconstruction;
  test_time_dependent_ad;
  test_viscosity
  )

# Add unit tests
FOREACH( sourcefile ${HHO_UNIT_TESTS} )
  ADD_EXECUTABLE(${sourcefile} ${sourcefile}.cpp)
  TARGET_LINK_LIBRARIES(${sourcefile}
      Mesh
      ${Boost_FILESYSTEM_LIBRARY_RELEASE}
      ${Boost_SYSTEM_LIBRARY_RELEASE}
      )
ENDFOREACH( sourcefile ${HHO_UNIT_TESTS} )
