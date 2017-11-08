PROJECT(hho_peaceman)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

IF(CMAKE_SYSTEM_NAME=Windows)
  SET(ADDITIONAL_FLAGS "-DWIN32")
ENDIF(CMAKE_SYSTEM_NAME=Windows)

SET(CMAKE_CXX_COMPILER "clang++")
SET(CMAKE_CXX_FLAGS_DEBUG "-std=c++14 -g -Wall -DVF_DEBUG" ${ADDITIONAL_FLAGS} CACHE STRING "Debug flags" FORCE)
SET(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wall -O3 -DNDEBUG -DOPTIMIZE" ${ADDITIONAL_FLAGS} CACHE STRING "Release flags" FORCE)
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-std=c++14 -Wall -O3 -g -DNDEBUG -DOPTIMIZE" ${ADDITIONAL_FLAGS} CACHE STRING "Release with debug info flags" FORCE)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(HOME_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
SET(CMAKE_MODULE_PATH ${HOME_DIR}/CMake)

SET(Boost_DEBUG "ON")
SET(Boost_USE_STATIC_LIBS "ON")
FIND_PACKAGE(Boost COMPONENTS filesystem program_options REQUIRED)
FIND_PACKAGE(Eigen3 3.3)
SET(HHO_INCLUDE_LIST ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

# Include libraries
INCLUDE_DIRECTORIES(${HHO_INCLUDE_LIST})
INCLUDE_DIRECTORIES(src)

# Code to build
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(UnitTests)
ADD_SUBDIRECTORY(PeacemanTests)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#



