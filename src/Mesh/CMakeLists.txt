# CMake configuration for the mesh library

ADD_LIBRARY(Mesh
  mapped_file.cpp
  util.cpp
  contrib/triangle_dunavant_rule.cpp
)
