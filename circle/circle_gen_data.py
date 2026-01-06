# gen_data.py
#
# Usage: python3 gen_data.py /path/to/cfg.json /path/to/logdir/
#  Reads a configuration JSON file and writes dataset NPY files to logdir
# Parameters:
#  /path/to/cfg.json: specifies amplitudes, frequencies, and phases in degrees
#  /path/to/logdir/: the path to the log directory for dataset files
# Output:
#  A dataset of NPY files

# import Python modules
import itertools   # product
import json        # JSON
import math        # ceil
import numpy as np # numpy
import os          # path.join
import sys         # argv

# "constants"
R = 0 # index for radius
V = 1 # index for vertical offset
H = 2 # index for horizontal offset

# helper functions
## None

# initialize script arguments
cfg = '' # path to configuration file
log = '' # path to log file

# parse script arguments
if len(sys.argv)==3:
  cfg = sys.argv[1]
  log = sys.argv[2]
else:
  print(\
   'Usage: '\
   'python3 gen_data.py /path/to/cfg.json /path/to/logdir/'\
  )
  exit()

# load configuration parameters
radii = []
vertical_offsets = []
horizontal_offsets = []
with open(cfg, 'r') as ifile:
  json_dict = json.load(ifile)
  radii = json_dict['radius']
  vertical_offsets = json_dict['vertical_offset']
  horizontal_offsets = json_dict['horizontal_offset']

# determine number of steps
steps = 1000

# generate all wave combinations
circles = list(itertools.product(radii, vertical_offsets, horizontal_offsets))

# determine zfill padding
pad = math.floor(math.log10(len(circles))) + 1

id_to_cfg = {}
circle_id = 0
for circle in circles:
  circle_id_str = str(circle_id).zfill(pad)
  r = circle[R]
  v = circle[V]
  h = circle[H]
  id_to_cfg[circle_id_str] = {
   'radius': r,
   'vertical_offset': v,
   'horizontal_offset': h
  }
  
  # determine the minimum x-axis window size for all circles to be complete
  x_max = h + r

  # determine the minimum y-axis window size for all circles to be complete
  x_min = h - r

  '''
  # for each circle, write x_values
  x_values = np.linspace(x_min, x_max, steps)

  # equation for circle
  inside = r**2 - (x_values-h)**2 
  inside = np.maximum(inside, 0.0) 
  y_pos = v + np.sqrt(inside)
  y_neg = v - np.sqrt(inside)
  '''

  theta = np.linspace(0, 2*np.pi, steps)
  x_values_full = h + r*np.cos(theta)
  y_values_full = v + r*np.sin(theta)
 
  '''
  # combine positive and negative halves
  y_values_full = np.concatenate((y_pos, y_neg))
  x_values_full = np.concatenate((x_values, x_values))
  '''

  # Generates circle data with x values, y values, and corresponding theta values
  circle_out = np.column_stack((x_values_full, y_values_full, theta)) 
  np.save(os.path.join(log, circle_id_str+'.npy'), circle_out)
  circle_id += 1

# write a JSON configuration key
with open(os.path.join(log,'circle-npy-to-cfg.json'), 'w') as ofile:
  json.dump(id_to_cfg, ofile)
