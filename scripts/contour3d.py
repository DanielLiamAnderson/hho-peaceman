#!/usr/bin/python3

# ---------------------------------------------------------------
#   Generates contour plots based on a set of (x,y,z) triples.
#   Input can be from VTU files, VTK files or raw triples.
#
#   Usage:
#      contour3d.py filename.vtu
# ---------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm

# -----------------------------------------------------------------
#                Set up command line arguments

parser = argparse.ArgumentParser(description='Generate contour plots from VTU or raw data files.')

parser.add_argument('filename', type=str, help="the file containing the plot.")

# Parse arguments
args = parser.parse_args()

# Extract points from vtu file
if ('.vtu' in args.filename):
  data = [line.rstrip('\n') for line in open(args.filename)]
  # Chop off the sections of the file we don't care about
  data = data[(data.index('<DataArray type="Float32" NumberOfComponents="3" format="ascii">')+1):]
  data = data[:data.index('</DataArray>')]
  x = [float(line.split()[0]) for line in data]
  y = [float(line.split()[1]) for line in data]
  z = [float(line.split()[2]) for line in data]
# Extract points from vtk file
elif ('.vtk' in args.filename):
   data = [line.rstrip('\n') for line in open(args.filename)]
   points_header = next(line for line in data if "points" in line.lower())
   begin = data.index(points_header)+1
   num_pts = int(points_header.split()[1])
   data = data[begin:begin+num_pts]
   x = [float(line.split()[0]) for line in data]
   y = [float(line.split()[1]) for line in data]
   z = [float(line.split()[2]) for line in data]
# Assume that anything else is just a list of triples
else:
  data = np.genfromtxt(args.filename)
  x = data[:,0]
  y = data[:,1]
  z = data[:,2]

# Remove duplicate x/y values
cmap = {}
for i in range(len(x)):
  if (x[i], y[i]) in cmap:
    cmap[(x[i], y[i])].append(z[i])
  else:
    cmap[(x[i], y[i])] = [z[i]]
    
x = []
y = []
z = []
for p in cmap:
  x.append(p[0])
  y.append(p[1])
  z.append(np.mean(cmap[p]))
  
# Interpolate grid data
xi = np.linspace(min(x), max(x))
yi = np.linspace(min(y), max(y))
zi = griddata(x, y, z, xi, yi, interp='linear')

# Plot a contour plot
fig = plt.figure()
CS = plt.contour(xi, yi, zi)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()

