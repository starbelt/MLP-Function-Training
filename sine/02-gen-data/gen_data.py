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
A = 0 # index for amplitude
F = 1 # index for frequency
P = 2 # index for phase

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
amplitudes = []
frequencies = []
phases = []
with open(cfg, 'r') as ifile:
  json_dict = json.load(ifile)
  amplitudes = json_dict['amplitude']
  frequencies = json_dict['frequency']
  phases = np.array(json_dict['phase-deg'])*np.pi/180.0
  phases = phases.tolist()

# determine the minimum time for all waves to have three periods
t_max = 3.0/min(frequencies)

# determine the maximum step duration to meet engineer's nyquist
t_stp_max = 1.0/(2.2*max(frequencies))

# determine number of steps
steps = max(math.ceil(t_max/t_stp_max),10000)

# generate all wave combinations
waves = list(itertools.product(amplitudes,frequencies,phases))

# determine zfill padding
pad = math.floor(math.log10(len(waves)))+1

# for each wave, write out a dataset
ts = np.linspace(0.0,t_max,steps)
id_to_cfg = {}
wave_id = 0
for wave in waves:
  wave_id_str = str(wave_id).zfill(pad)
  a = wave[A]
  f = wave[F]
  p = wave[P]
  id_to_cfg[wave_id_str] = {
   'amplitude': a,
   'frequency': f,
   'phase-rad': p
  }
  ys = a*np.sin(2.0*np.pi*f*ts+p)
  wave_out = np.column_stack((ts,ys))
  np.save(os.path.join(log,wave_id_str+'.npy'),wave_out)
  wave_id += 1

# write a JSON configuration key
with open(os.path.join(log,'npy-to-cfg.json'), 'w') as ofile:
  json.dump(id_to_cfg,ofile)
