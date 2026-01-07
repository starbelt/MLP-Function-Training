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
A = 0 # index for a term
B = 1 # index for b term
C = 2 # index for c term

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
a_terms = []
b_terms = []
c_terms = []
with open(cfg, 'r') as ifile:
  json_dict = json.load(ifile)
  a_terms = json_dict['a']
  b_terms = json_dict['b']
  c_terms = json_dict['c']

# determine number of steps
steps = 1000

# generate all parabola combinations
parabolas = list(itertools.product(a_terms,b_terms,c_terms))

# determine zfill padding
pad = math.floor(math.log10(len(parabolas)))+1

# for each parabola, write out a dataset
x = np.linspace(-10.0,10.0, steps)
id_to_cfg = {}
parabola_id = 0
for parabola in parabolas:
  parabola_id_str = str(parabola_id).zfill(pad)
  a = parabola[A]
  b = parabola[B]
  c = parabola[C]
  id_to_cfg[parabola_id_str] = {
   'a': a,
   'b': b,
   'c': c
  }
  y = a*x**2 + b*x + c
  parabola_out = np.column_stack((x,y))
  np.save(os.path.join(log,parabola_id_str+'.npy'),parabola_out)
  parabola_id += 1

# write a JSON configuration key
with open(os.path.join(log,'npy-to-cfg.json'), 'w') as ofile:
  json.dump(id_to_cfg,ofile)
