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
with open(os.path.join(src,'npy-to-cfg.json'), 'r') as ifile:
  npy_to_cfg_dict = json.load(ifile)

# load data and write plot PNGs
for npy in npys:
  cap_id = npy[:-4]
  plt_title = \
   'Surface Area: '+'{:.3f}'.format(npy_to_cfg_dict[cap_id]['surface area'])+'; '+\
   'Capacitance: '+'{:.3f}'.format(npy_to_cfg_dict[cap_id]['capacitance'])+'; '+\
   'ESR: '+'{:.3f}'.format(npy_to_cfg_dict[cap_id]['equivalent series resistance'])+'; '+\
   'Power: '+'{:.3f}'.format(npy_to_cfg_dict[cap_id]['power'])
  plt_y_axis = 'Voltage (V)'
  plt_x_axis = 'Time (s)'
  nparr = np.load(os.path.join(src,npy))
  #fig = plt.figure(figsize=(8.5,4.0))
  fig = plt.figure(layout='constrained')
  plt.plot(nparr[:,0],nparr[:,1],marker='.',linestyle='None')
  plt.title(plt_title)
  plt.ylabel(plt_y_axis)
  plt.xlabel(plt_x_axis)
  plt.savefig(os.path.join(dst,cap_id+'.png'),format='png')
  plt.close(fig)
