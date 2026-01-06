# viz_data.py
#
# Usage: python3 viz_data.py /path/to/src/ /path/to/dst/
#  Reads all NPY files in src and writes PNG files to dst
# Parameters:
#  /path/to/src/: a directory containing NPY files
#  /path/to/dst/: a directory for writing PNG files
# Output:
#  A collection of PNG visualizations

# import Python modules
import json                     # json
import matplotlib.pyplot as plt # matplotlib
import numpy as np              # numpy
import os                       # listdir
import sys                      # argv

# "constants"
## None

# helper functions
## None

# initialize script arguments
src = '' # a directory containing NPY files
dst = '' # a directory for writing PNG files

# parse script arguments
if len(sys.argv)==3:
  src = sys.argv[1]
  dst = sys.argv[2]
else:
  print(\
   'Usage: '\
   'python3 viz_data.py /path/to/src/ /path/to/dst/'\
  )
  exit()

# collect NPY file paths
npys = [f for f in os.listdir(src) if f.endswith('.npy')]

# load the npy-to-cfg key to populate titles
npy_to_cfg_dict = {}
with open(os.path.join(src,'circle-npy-to-cfg.json'), 'r') as ifile:
  npy_to_cfg_dict = json.load(ifile)

# find max y value for consistent axis limits
max_y = 0.0
for k in npy_to_cfg_dict:
  k_r = npy_to_cfg_dict[k]['radius']
  k_v = npy_to_cfg_dict[k]['vertical_offset']
  if k_r + k_v > max_y:
    max_y = k_r + k_v
# find max x value for consistent axis limits
max_x = 0.0
for k in npy_to_cfg_dict:
  k_r = npy_to_cfg_dict[k]['radius']
  k_h = npy_to_cfg_dict[k]['horizontal_offset']
  if k_r + k_h > max_x:
    max_x = k_r + k_h

# load data and write plot PNGs
for npy in npys:
  wave_id = npy[:-4]
  plt_title = \
   'Radius: '+'{:.3f}'.format(npy_to_cfg_dict[wave_id]['radius'])+'; '+\
   'Vertical Offset: '+'{:.3f}'.format(npy_to_cfg_dict[wave_id]['vertical_offset'])+'; '+\
   'Horizontal Offset: '+'{:.3f}'.format(npy_to_cfg_dict[wave_id]['horizontal_offset'])
  plt_y_axis = 'Y'
  plt_x_axis = 'X'
  nparr = np.load(os.path.join(src,npy))
  #fig = plt.figure(figsize=(8.5,4.0))
  fig = plt.figure(layout='constrained')
  plt.plot(nparr[:,0],nparr[:,1],marker='.',linestyle='None')
  plt.axis("equal")
  plt.ylim(-0.5*max_y, 1.5*max_y)
  plt.xlim(-0.5*max_x, 1.1*max_x)
  plt.title(plt_title)
  plt.ylabel(plt_y_axis)
  plt.xlabel(plt_x_axis)
  plt.savefig(os.path.join(dst,wave_id+'.png'),format='png')
  plt.close(fig)
