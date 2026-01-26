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
from datetime import datetime
import time

# "constants"
EPOCHS = 50
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

def count_hidden_layers(json_dict):
    num_linear = sum(
        1 for layer in json_dict['layers']
        if layer['class'] == 'Linear'
    )
    # assume last Linear is output layer
    return max(0, num_linear - 1)

## computes mean and stddev of a tensor along dimension 0 (not original code)
def compute_mean_std(tensor, eps=1e-8):
  mean = tensor.mean(dim=0)
  std  = tensor.std(dim=0)
  std = torch.clamp(std, min=eps)
  return mean, std

# helper classes (original code kept for reference)
'''
## custom dataset
class MSDataset(Dataset):
  # src_dir contains npy-to-cfg.json and .npy files
  def __init__(self, src_dir):
    # read npy-to-cfg.json
    npy_to_cfg_dict = {}
    with open(os.path.join(src_dir,'npy-to-cfg.json'), 'r') as ifile:
      npy_to_cfg_dict = json.load(ifile)
    # iteratively read .npy files and add to data and label
    npys = [f for f in os.listdir(src_dir) if f.endswith('.npy')]
    for npy in npys:
      wave_id = npy[:-4]
      data_cfg = np.array([
       npy_to_cfg_dict[wave_id]['amplitude'],
       npy_to_cfg_dict[wave_id]['frequency'],
       npy_to_cfg_dict[wave_id]['phase-rad']
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
  def __init__(self, src_dir, t_mean=None, t_std=None,
               v_mean=None, v_std=None):
    with open(os.path.join(src_dir,'npy-to-cfg.json'), 'r') as ifile:
      npy_to_cfg_dict = json.load(ifile)

    npys = [f for f in os.listdir(src_dir) if f.endswith('.npy')]

    data_list = []
    label_list = []

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

      nparr = np.load(os.path.join(src_dir, npy))

      t = np.column_stack((
        np.repeat([data_cfg], repeats=nparr.shape[0], axis=0),
        nparr[:, 0]
      ))
      v = nparr[:, [1]]

      data_list.append(torch.tensor(t, dtype=torch.float32))
      label_list.append(torch.tensor(v, dtype=torch.float32))

    # concatenate all waves into one big dataset
    self.data = torch.cat(data_list, dim=0)
    self.labels = torch.cat(label_list, dim=0)

    # normalization code (not in original)
    self.t_mean = t_mean
    self.t_std = t_std
    self.v_mean = v_mean
    self.v_std = v_std


  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):

    # normalization code (not in original)
    t = self.data[idx]
    v = self.labels[idx]

    if self.t_mean is not None:
      t = (t - self.t_mean) / self.t_std
    if self.v_mean is not None:
      v = (v - self.v_mean) / self.v_std

    return t, v
# End of new code 

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

t_start_total = time.perf_counter()

# load JSON configuration of MLP
json_dict = {}
with open(cfg, 'r') as ifile:
  json_dict = json.load(ifile)

# get MLP cfg file name
mlp_id = os.path.splitext(os.path.basename(cfg))[0]
#print(mlp_id)

# create specified MLP model
mlp = mlp_from_json(json_dict)

# criterion: use a regression loss function, specifically MSE
criterion = nn.MSELoss()

# optimizer: adaptive moment estimation to automatically adjust learning rate
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

# load datasets
trn_dataset_raw = MSDataset(src_dir=os.path.join(src,'trn'))
#val_dataset_raw = MSDataset(src_dir=os.path.join(src,'val'))

t_mean, t_std = compute_mean_std(trn_dataset_raw.data)
v_mean, v_std = compute_mean_std(trn_dataset_raw.labels)

trn_dataset = MSDataset(src_dir=os.path.join(src,'trn'),
                        t_mean=t_mean, t_std=t_std,
                        v_mean=v_mean, v_std=v_std)

val_dataset = MSDataset(src_dir=os.path.join(src,'val'),
                        t_mean=t_mean, t_std=t_std,
                        v_mean=v_mean, v_std=v_std)

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


run_dir = os.path.join(dst, f"{mlp_id}")
os.makedirs(run_dir, exist_ok=True)
print(run_dir)

# write losses to CSV file
with open(os.path.join(run_dir, f"{mlp_id}-losses.csv"),mode='w',newline='') as ofile:
  csvwriter = csv.writer(ofile)
  csvwriter.writerow(losses[0])
  for row in losses[1:]:
    csvwriter.writerow(\
     [int(row[0]), '{:.9f}'.format(row[1]), '{:.9f}'.format(row[2])]\
    )

# save model file
torch.save(mlp.state_dict(), os.path.join(run_dir,f"{mlp_id}.pt"))

torch.save({
  "t_mean": t_mean,
  "t_std": t_std,
  "v_mean": v_mean,
  "v_std": v_std
}, os.path.join(run_dir, f"{mlp_id}-norm.pt"))

t_end_total = time.perf_counter()
total_time_s = t_end_total - t_start_total

runtime_csv = os.path.join(run_dir, f"{mlp_id}-runtime.csv")

mse_norm = losses[-1][2] 
mse_volts2 = mse_norm * float(v_std.item() ** 2) 
rmse_volts = (mse_norm ** 0.5) * float(v_std.item())

with open(runtime_csv, mode='w', newline='') as ofile:
    writer = csv.writer(ofile)
    writer.writerow(['metric', 'seconds'])
    writer.writerow(['total_execution_time', f'{total_time_s:.6f}'])
    writer.writerow(['final_val_rmse', f'{rmse_volts:.9f}'])