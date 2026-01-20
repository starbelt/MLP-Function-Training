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
with open(os.path.join(src,'npy-to-cfg.json'), 'r') as ifile:
  npy_to_cfg_dict = json.load(ifile)

# load data and write plot PNGs
for npy in npys:
  cap_id = npy[:-4]
  data_cfg = np.array([
    npy_to_cfg_dict[cap_id]['surface area'],
    npy_to_cfg_dict[cap_id]['efficiency'],
    npy_to_cfg_dict[cap_id]['max power voltage'],
    npy_to_cfg_dict[cap_id]['capacitance'],
    npy_to_cfg_dict[cap_id]['equivalent series resistance'],
    npy_to_cfg_dict[cap_id]['initial charge'],
    npy_to_cfg_dict[cap_id]['power'],
    npy_to_cfg_dict[cap_id]['high voltage'],
    npy_to_cfg_dict[cap_id]['low voltage'],
  ])
  nparr = np.load(os.path.join(src,npy))
  inputs = torch.tensor(np.column_stack((\
   np.repeat([data_cfg],repeats=nparr.shape[0],axis=0),nparr[:,0]\
  )),dtype=torch.float32)
  pred_out = mlp(inputs)

  # Test section: Not in original code

  with torch.no_grad():
    pred_out = mlp(inputs)

  pred_np = pred_out.cpu().numpy().squeeze()   # (N,)
  true_np = nparr[:, 1]                        # (N,)

  mse = np.mean((pred_np - true_np) ** 2)
  mae = np.mean(np.abs(pred_np - true_np))
  max_err = np.max(np.abs(pred_np - true_np))

  print(f"{cap_id}: MSE={mse:.6f}, MAE={mae:.6f}, MaxErr={max_err:.6f}")

  plt_title = \
   'Surface Area: '+'{:.3f}'.format(npy_to_cfg_dict[cap_id]['surface area'])+'; '+\
   'Capacitance: '+'{:.3f}'.format(npy_to_cfg_dict[cap_id]['capacitance'])+'; '+\
   'ESR: '+'{:.3f}'.format(npy_to_cfg_dict[cap_id]['equivalent series resistance'])+'; '+\
   'Power: '+'{:.3f}'.format(npy_to_cfg_dict[cap_id]['power'])
  plt_y_axis = 'Voltage (V)'
  plt_x_axis = 'Time (s)'
  fig = plt.figure(layout='constrained')
  plt.plot(\
   nparr[:,0], nparr[:,1],\
   marker='o', linestyle='None', label='Truth'\
  )
  pred_np = pred_out.detach().cpu().numpy().squeeze()  # shape (N,)
  plt.plot(
   nparr[:,0], pred_np,
   marker='.', linestyle='None', label='Predictions'
  )
  '''
  plt.plot(\
   nparr[:,0], nparr[:,1],\
   marker='.', linestyle='None', label='Predictions'\
  )
  '''
  plt.title(plt_title)
  plt.ylabel(plt_y_axis)
  plt.xlabel(plt_x_axis)
  plt.legend()
  plt.savefig(os.path.join(dst,cap_id+'.png'),format='png')
  plt.close(fig)
