# trn_val_mlp.py
#
# Usage: python3 trn_val_mlp.py /path/to/mlp-cfg.json /path/to/src /path/to/dst
#  Generates MLP defined by mlp-cfg.json, trains and validates on src/trn and
#  src/val, and writes the trained model to dst
# Parameters:
#  /path/to/mlp-cfg.json: a JSON specification of an MLP
#  /path/to/src/: a directory containing trn/ and val/ directories
#  /path/to/dst/: a directory to write the trained MLP and performance stats
# Output:
#  A trained MLP model and performance statistics

# import Python modules
import csv            # csv writer
import json           # json
import numpy as np    # numpy
import os             # listdir
import sys            # argv
import torch          # PyTorch
import torch.nn as nn # Sequential, Linear, ReLU
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# "constants"
EPOCHS = 100
NUM_CPUS = os.cpu_count()

# helper functions

## accepts a JSON dictionary (the result of son.load(ifile)) as input
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

# helper classes
'''
## custom dataset
class MSDataset(Dataset):
  # src_dir contains npy-to-cfg.json and .npy files
  def __init__(self, src_dir):
    # read npy-to-cfg.json
    npy_to_cfg_dict = {}
    with open(os.path.join(src_dir,'circle-npy-to-cfg.json'), 'r') as ifile:
      npy_to_cfg_dict = json.load(ifile)
    # iteratively read .npy files and add to data and label
    npys = [f for f in os.listdir(src_dir) if f.endswith('.npy')]
    for npy in npys:
      circle_id = npy[:-4]
      data_cfg = np.array([
       npy_to_cfg_dict[circle_id]['radius'],
       npy_to_cfg_dict[circle_id]['vertical_offset'],
       npy_to_cfg_dict[circle_id]['horizontal_offset']
      ])
      nparr = np.load(os.path.join(src_dir,npy))
      self.data = torch.tensor(np.column_stack((\
       np.repeat([data_cfg],repeats=nparr.shape[0],axis=0),nparr[:,0]\
      )),dtype=torch.float32)
      self.labels = torch.tensor(nparr[:,[1]],dtype=torch.float32)

  # src_dir contains npy-to-cfg.json and .npy files
  def __len__(self):
    return len(self.data)

  # src_dir contains npy-to-cfg.json and .npy files
  def __getitem__(self, idx):
    sample = self.data[idx]
    label = self.labels[idx]
    return sample, label
'''
# New code for testing
class MSDataset(Dataset):
  def __init__(self, src_dir, R_MAX, V_OFF_MAX, H_OFF_MAX, XY_MAX):
    with open(os.path.join(src_dir,'npy-to-cfg.json'), 'r') as ifile:
      npy_to_cfg_dict = json.load(ifile)

    npys = [f for f in os.listdir(src_dir) if f.endswith('.npy')]

    data_list = []
    label_list = []

    for npy in npys:
      circle_id = npy[:-4]
      cfg = np.array([
        npy_to_cfg_dict[circle_id]['radius'],
        npy_to_cfg_dict[circle_id]['vertical_offset'],
        npy_to_cfg_dict[circle_id]['horizontal_offset']
      ], dtype=np.float32)

      nparr = np.load(os.path.join(src_dir, npy))

      
      r_n = cfg[0] / R_MAX
      v_n = cfg[1] / V_OFF_MAX
      h_n = cfg[2] / H_OFF_MAX

      cfg_norm = np.array([r_n, v_n, h_n], dtype=np.float32)

      nparr = np.load(os.path.join(src_dir, npy)).astype(np.float32)

      # Debugging print statement
      #print(npy, nparr.shape)

      sin_t = nparr[:, [2]]
      cos_t = nparr[:, [3]]
      xy = nparr[:, 0:2] / XY_MAX # normalize x and y
      
      cfg_rep = np.repeat(cfg_norm[None, :], repeats=nparr.shape[0], axis=0)  # (N,3)
      inputs_np = np.hstack([cfg_rep, sin_t, cos_t])                            # (N,5) -> [r, v_off, h_off, sin(theta), cos(theta)]

      data_list.append(torch.tensor(inputs_np, dtype=torch.float32))
      label_list.append(torch.tensor(xy, dtype=torch.float32))

    # concatenate all waves into one big dataset
    self.data = torch.cat(data_list, dim=0)
    self.labels = torch.cat(label_list, dim=0)

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
# end of new code 

# initialize script arguments
cfg = '' # a JSON specification of an MLP
src = '' # a directory containing trn/ and val/ directories
dst = '' # a directory to write the trained MLP and performance stats

# parse script arguments
if len(sys.argv)==4:
  cfg = sys.argv[1]
  src = sys.argv[2]
  dst = sys.argv[3]
else:
  print(\
   'Usage: '\
   'python3 trn_val_mlp.py /path/to/mlp-cfg.json /path/to/src /path/to/dst'\
  )
  exit()

# load JSON configuration of MLP
json_dict = {}
with open(cfg, 'r') as ifile:
  json_dict = json.load(ifile)

# get MLP cfg file name
mlp_id = os.path.splitext(os.path.basename(cfg))[0]

# create specified MLP model
mlp = mlp_from_json(json_dict)

# criterion: use a regression loss function, specifically MSE
criterion = nn.MSELoss()

# optimizer: adaptive moment estimation to automatically adjust learning rate
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

# Compute normalization constants ONCE from training config
with open(os.path.join(src, 'trn', 'npy-to-cfg.json'), 'r') as f:
  trn_cfg_dict = json.load(f)

R_MAX = max(trn_cfg_dict[k]['radius'] for k in trn_cfg_dict)
V_OFF_MAX = max(trn_cfg_dict[k]['vertical_offset'] for k in trn_cfg_dict)
H_OFF_MAX = max(trn_cfg_dict[k]['horizontal_offset'] for k in trn_cfg_dict)

R_MAX = max(R_MAX, 1.0)
V_OFF_MAX = max(V_OFF_MAX, 1.0)
H_OFF_MAX = max(H_OFF_MAX, 1.0)

XY_MAX = max(R_MAX + H_OFF_MAX, R_MAX + V_OFF_MAX, 1.0)

# load datasets
trn_dataset = MSDataset(os.path.join(src,'trn'), R_MAX, V_OFF_MAX, H_OFF_MAX, XY_MAX)
val_dataset = MSDataset(os.path.join(src,'val'), R_MAX, V_OFF_MAX, H_OFF_MAX, XY_MAX)

# construct data loaders
worker_count = min(NUM_CPUS,16) # no need for more than 16 data loader workers
trn_loader = DataLoader(\
 dataset=trn_dataset, batch_size=512, shuffle=True, num_workers=worker_count\
)
val_loader = DataLoader(\
 dataset=val_dataset, batch_size=1024, shuffle=True, num_workers=worker_count\
)

# train and validate
losses = [['epoch','trn','val']]
for i in tqdm(range(0,EPOCHS), desc='Performing trn and val epochs'):
  ## train
  mlp.train() # ensure training-specific layers are active if present
  trn_loss = 0.0
  for inputs, true_out in trn_loader:
    optimizer.zero_grad()  # clear gradients
    pred_out = mlp(inputs) # forwards pass
    loss = criterion(pred_out, true_out)
    loss.backward()        # backward pass
    optimizer.step()       # update weights
    trn_loss += loss.item()
  ## validate
  mlp.eval() # disable any training-specific layers if present
  val_loss = 0.0
  with torch.no_grad():
    for inputs, true_out in val_loader:
      pred_out = mlp(inputs)
      loss = criterion(pred_out, true_out)
      val_loss += loss.item()
  ## store updates
  losses.append([i,trn_loss/len(trn_loader),val_loss/len(val_loader)])
  ## print updates
  #print('Epoch '+str(i).zfill(3)+':')
  #print('  Trn loss: {:.6f}'.format(trn_losses[-1][1]))
  #print('  Val loss: {:.6f}'.format(val_losses[-1][1]))

# write losses to CSV file
with open(os.path.join(dst,mlp_id+'-losses.csv'),mode='w',newline='') as ofile:
  csvwriter = csv.writer(ofile)
  csvwriter.writerow(losses[0])
  for row in losses[1:]:
    csvwriter.writerow(\
     [int(row[0]), '{:.9f}'.format(row[1]), '{:.9f}'.format(row[2])]\
    )

# save model file
torch.save(mlp.state_dict(), os.path.join(dst,mlp_id+'.pt'))
