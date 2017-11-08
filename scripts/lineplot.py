#!/usr/bin/python3

# ---------------------------------------------------------------
#   Generates log-log plots of convergence rates from
#   error data produced by the scheme test programs.
#
#   Use the --no-title option to suppress drawing a title. Use
#   --xlabel to override the default x axis label, and similarly
#   --ylabel. Add --slope to draw slope indicators for analysing
#   the order of convergence.
#
#   Usage:
#      lineplot.py [options] filename
#
# ---------------------------------------------------------------

# File structure is as follows
# TITLE
# XLABEL
# YLABEL
# x-values
# For each scheme:
#    label
#    y-values

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

from mpltools import annotation
from math import log

# -----------------------------------------------------------------
#                               Default options

# Markers / Colours for different degree schemes k = 0,1,2,3,4,5
markers = ['.', 'o', '^', 's', 'p', 'h', '8']
colours = ['g', 'c', 'b', '#cc00cc', '#ff6600', 'r', '#000000']

# -----------------------------------------------------------------
#                           Set up command line arguments

parser = argparse.ArgumentParser(description='Generate log-log plots of convergence data.')
parser.add_argument('filename', type=str, help="the file containing convergence data.")
parser.add_argument('-n', '--no-title', action="store_true", dest="notitle", help="don't display a title on the plot")
parser.add_argument('-s', '--slope', action="store_true", dest="slope", help="Display a slope indicator")
parser.add_argument('-x', '--xlabel', action="store", dest="xlabel", help="Override the x-axis label present in the data file")
parser.add_argument('-y', '--ylabel', action="store", dest="ylabel", help="Override the y-axis label in the data file")
parser.set_defaults(notitle=False)

# Parse arguments
args = parser.parse_args()

# -----------------------------------------------------------------
#                           Read and parse data

# Read data from file
data = [line.rstrip('\n') for line in open(args.filename)]

# Read the title and labels
title = data[0]
xlabel = data[1]
ylabel = data[2]

# Override labels
if not args.xlabel == None:
	xlabel = args.xlabel
if not args.ylabel == None:
	ylabel = args.ylabel

# Read the mesh sizes
meshsize = [float(h) for h in data[3].split()]

numschemes = (len(data) - 4) // 2

# Read errors for each scheme
scheme_name = ["..." for k in range(numschemes)]
errors = [[] for k in range(numschemes)]
for k in range(numschemes):
  scheme_name[k] = data[4 + 2 * k]
  errors[k] = [float(e) for e in data[4 + 2 * k + 1].split()]

# Configure plots
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xscale('log')
plt.yscale('log')
plt.title('' if args.notitle else title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.autoscale()
  
font = {'family' : 'normal',
      'weight' : '500',
      'size'   : 18}

matplotlib.rc('font', **font)
  
# Make a plot for each scheme
lines = []
for k in range(0, numschemes):
  lines.append(ax.plot(meshsize, errors[k],
    linestyle='-',
    marker=markers[k],
    color=colours[k],
    label=scheme_name[k]
    )
  )
  # Compute slope
  if (args.slope):
    slopes = []
    for i in range(len(meshsize) - 1):
      slopes.append((log(errors[k][i + 1]) - log(errors[k][i])) / (log(meshsize[i + 1]) - log(meshsize[i])))
    print(slopes)
    print(np.std(slopes))
    if (np.std(slopes) < 0.5):
      slope = round(slopes[-1], 2)
      pos = (meshsize[-1], errors[k][-1])
      annotation.slope_marker(pos, slope, ax=ax, invert=False, poly_kwargs={'color': colours[k]})
  
plt.legend()
plt.tight_layout()
plt.show()

