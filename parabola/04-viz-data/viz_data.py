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

# find max y value
max_y = -np.inf

for fname in os.listdir(src):
    if not fname.endswith(".npy"):
        continue

    arr = np.load(os.path.join(src, fname))
    y_vals = arr[:, 1]          # y is column 1
    max_y = max(max_y, y_vals.max())

# load data and write plot PNGs
for npy in npys:
  wave_id = npy[:-4]
  plt_title = \
   'A-term: '+'{:.3f}'.format(npy_to_cfg_dict[wave_id]['a'])+'; '+\
   'B-term: '+'{:.3f}'.format(npy_to_cfg_dict[wave_id]['b'])+'; '+\
   'C-term: '+'{:.3f}'.format(npy_to_cfg_dict[wave_id]['c'])
  plt_y_axis = 'f(x)'
  plt_x_axis = 'x'

  nparr = np.load(os.path.join(src,npy))
  #fig = plt.figure(figsize=(8.5,4.0))
  fig = plt.figure(layout='constrained')
  plt.plot(nparr[:,0],nparr[:,1],marker='.',linestyle='None')
  plt.ylim(-1.1*max_y, 1.1*max_y)
  plt.title(plt_title)
  plt.ylabel(plt_y_axis)
  plt.xlabel(plt_x_axis)
  plt.savefig(os.path.join(dst,wave_id+'.png'),format='png')
  plt.close(fig)
