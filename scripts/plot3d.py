#!/usr/bin/python3

# ---------------------------------------------------------------
#   Generates surface plots based on a set of (x,y,z) triples.
#   Input can be from VTU files, VTK files or raw triples.
#   Can add --edge-colour option to set the colours of the
#   edges of the surface triangulation.
#
#   Usage:
#      plot3d.py [options] filename.vtu
# ---------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm
from random import random

# -----------------------------------------------------------------
#                Set up command line arguments

parser = argparse.ArgumentParser(description='Generate 3d surface plots from VTU or raw data files.')

parser.add_argument('filename', type=str, help="the file containing the plot.")
parser.add_argument('-e', '--edge-colour', action="store", dest="edgecolour", default="none", help="edge colour for the triangulation, or (none)")

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

# Assume that non-vtu files are just lists of triples
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

# Sample some data points if there are too many
threshold = 2500
if (len(x) > threshold):
  print("There are {:d} data points, samping {:d} of them".format(len(x), threshold))
  nx, ny, nz = [], [], []
  p = threshold / len(x)
  
  boundary = [np.min(x), np.max(x), np.min(y), np.max(y)]
  
  # Take points with probability p (always take boundary points)
  for i in range(len(x)):
    if (random() < p or x[i] in boundary or y[i] in boundary):
      nx.append(x[i])
      ny.append(y[i])
      nz.append(z[i])
  x, y, z = nx, ny, nz

# Plot the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_axis_bgcolor('white')
surf = ax.plot_trisurf(x, y, z, cmap = cm.jet, edgecolor=args.edgecolour) 
ax.set_zlim3d(min(0, 1.01 * np.min(z)), 1.01 * np.max(z))
ax.tick_params(axis=u'both', which=u'both',length=0)

#fig.colorbar(surf)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()

