# viz_mlp.py
#
# Usage: python3 viz_mlp.py /path/to/mlp-cfg.json /path/to/mlp.pt /path/to/src/ /path/to/dst/
#  Generates MLP defined by mlp-cfg.json, loads weights from mlp.pt, reads all
#  NPY files in src, and writes PNG files to dst
# Parameters:
#  /path/to/mlp-cfg.json: a JSON specification of an MLP
#  /path/to/mlp.pt: a PyTorch model weights file
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
import torch                    # PyTorch
import torch.nn as nn           # Sequential, Linear, ReLU

# "constants"
## None

# helper functions

## accepts a JSON dictionary (the result of json.load(ifile)) as input
## returns a PyTorch nn.Sequential MLP as output
def mlp_from_json(json_dict):
  in_features = json_dict['in_features']
  layers = []
  for layer_cfg in json_dict['layers']:
    layer_class = layer_cfg['class']
    if layer_class=='Linear':
      out_features = layer_cfg['out_features']
      layers.append(nn.Linear(in_features,out_features))
      in_features = out_features
    elif layer_class=='ReLU':
      layers.append(nn.ReLU())
    else:
      print('Layer class not yet implemented: '+layer_class)
      print('  Exiting')
      exit()
  return nn.Sequential(*layers)

# initialize script arguments
cfg = '' # a JSON specification of an MLP
pth = '' # a PyTorch model weights file
src = '' # a directory containing NPY files
dst = '' # a directory for writing PNG files

# parse script arguments
if len(sys.argv)==5:
  cfg = sys.argv[1]
  pth = sys.argv[2]
  src = sys.argv[3]
  dst = sys.argv[4]
else:
  print(\
   'Usage: '\
   'python3 viz_mlp.py /path/to/mlp-cfg.json /path/to/mlp.pt /path/to/src/ /path/to/dst/'\
  )
  exit()

# load JSON configuration of MLP
json_dict = {}
with open(cfg, 'r') as ifile:
  json_dict = json.load(ifile)

# create specified MLP model
mlp = mlp_from_json(json_dict)

# load state dictionary
state_dict = torch.load(pth)
mlp.load_state_dict(state_dict)
mlp.eval()

# collect NPY file paths
npys = [f for f in os.listdir(src) if f.endswith('.npy')]

# load the npy-to-cfg key to populate titles
npy_to_cfg_dict = {}
with open(os.path.join(src,'circle-npy-to-cfg.json'), 'r') as ifile:
  npy_to_cfg_dict = json.load(ifile)

# --- load TRAIN cfg dict to compute normalization constants (must match training!) ---
with open(os.path.join(src,'circle-npy-to-cfg.json'), 'r') as ifile:
  trn_cfg_dict = json.load(ifile)

R_MAX = max(trn_cfg_dict[k]['radius'] for k in trn_cfg_dict)
V_OFF_MAX = max(trn_cfg_dict[k]['vertical_offset'] for k in trn_cfg_dict)
H_OFF_MAX = max(trn_cfg_dict[k]['horizontal_offset'] for k in trn_cfg_dict)

R_MAX = max(R_MAX, 1.0)
V_OFF_MAX = max(V_OFF_MAX, 1.0)
H_OFF_MAX = max(H_OFF_MAX, 1.0)

XY_MAX = max(R_MAX + H_OFF_MAX, R_MAX + V_OFF_MAX, 1.0)

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
  circle_id = npy[:-4]
  data_cfg = np.array([
   npy_to_cfg_dict[circle_id]['radius'],
   npy_to_cfg_dict[circle_id]['vertical_offset'],
   npy_to_cfg_dict[circle_id]['horizontal_offset']
  ])
  nparr = np.load(os.path.join(src,npy))

  # columns: [x, y, theta]
  x_true = nparr[:, 0] 
  y_true = nparr[:, 1] 
  sin_t = nparr[:, [2]]  # (N,1)
  cos_t = nparr[:, [3]]  # (N,1)

  r = data_cfg[0]
  v = data_cfg[1]
  h = data_cfg[2] 

  cfg_norm = np.array([r / R_MAX, v / V_OFF_MAX, h / H_OFF_MAX], dtype=np.float32)
  cfg_rep = np.repeat(cfg_norm[None, :], repeats=nparr.shape[0], axis=0)  # (N,3)

  # inputs: [radius, vertical_offset, horizontal_offset, sin(theta), cos(theta)]
  inputs_np = np.hstack([cfg_rep, sin_t, cos_t])  # (N,5)
  inputs = torch.tensor(inputs_np, dtype=torch.float32)

  with torch.no_grad():
    pred_out = mlp(inputs)  # (N,2)

  plt_title = \
   'Radius: '+'{:.3f}'.format(npy_to_cfg_dict[circle_id]['radius'])+'; '+\
   'Vertical Offset: '+'{:.3f}'.format(npy_to_cfg_dict[circle_id]['vertical_offset'])+'; '+\
   'Horizontal Offset: '+'{:.3f}'.format(npy_to_cfg_dict[circle_id]['horizontal_offset'])
  plt_y_axis = 'Y Axis'
  plt_x_axis = 'X Axis'
  fig = plt.figure(layout='constrained')
  
  plt.plot(\
   x_true, y_true,
   marker='o', linestyle='None', label='Truth'\
  )
  pred_np = pred_out.cpu().numpy()  # (N,2)
  x_pred = pred_np[:, 0] * XY_MAX
  y_pred = pred_np[:, 1] * XY_MAX
  
  plt.plot(
   x_pred, y_pred,
   marker='.', linestyle='None', label='Predictions'
  )

  plt.axis("equal")
  plt.ylim(-max_y, 1.8*max_y)
  plt.xlim(-max_x, 1.1*max_x)
  plt.title(plt_title)
  plt.ylabel(plt_y_axis)
  plt.xlabel(plt_x_axis)
  plt.legend()
  plt.savefig(os.path.join(dst,circle_id+'.png'),format='png')
  plt.close(fig)

def list_circles(split):
  return sorted(f[:-4] for f in os.listdir(f"../05-split-data/{split}") if f.endswith(".npy"))

# print summary of circles in each split (only works if ../05-split-data/ is splitting individual circles into types)
'''
print("Train circles:", list_circles("trn"))
print("Val circles:  ", list_circles("val"))
print("Test circles: ", list_circles("tst")) 
'''