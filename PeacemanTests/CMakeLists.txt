# CMake configuration for HHO Peaceman tests
SET(MAX_DEGREE 3)

# Statistics tests
SET(HHO_PEACEMAN_TESTS
  test_computation_time;
  test_oil_volume;
  test_oil_volume_discontinuous;
  )

# Add statistics tests
FOREACH(sourcefile ${HHO_PEACEMAN_TESTS})
  ADD_EXECUTABLE(${sourcefile} ${sourcefile}.cpp)
  TARGET_LINK_LIBRARIES(${sourcefile}
      Mesh
      ${Boost_FILESYSTEM_LIBRARY_RELEASE}
      ${Boost_SYSTEM_LIBRARY_RELEASE}
      ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}
      )
ENDFOREACH(sourcefile ${HHO_UNIT_TESTS})

# Tests for Peaceman Model solver for degree k = 0...MAX_DEGREE
FOREACH(DEGREE RANGE 0 ${MAX_DEGREE})
  ADD_EXECUTABLE(test_peaceman${DEGREE} test_peaceman.cpp)
  TARGET_LINK_LIBRARIES(test_peaceman${DEGREE}
      Mesh
      ${Boost_FILESYSTEM_LIBRARY_RELEASE}
      ${Boost_SYSTEM_LIBRARY_RELEASE}
      ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}
      )
  SET_TARGET_PROPERTIES(test_peaceman${DEGREE} PROPERTIES COMPILE_FLAGS "-DDEGREE=${DEGREE}")
ENDFOREACH(DEGREE RANGE 0 ${MAX_DEGREE})
