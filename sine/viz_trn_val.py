# viz_trn_val.py
#
# Usage: python3 viz_trn_val.py /path/to/src/losses.csv /path/to/dst/
#  Reads src CSV and writes PDF visualization to dst
# Parameters:
#  /path/to/src/losses.csv: a header followed by (epoch, trn, val) rows
#  /path/to/dst/: a directory for writing visualization PDF
# Output:
#  A PDF visualization of training and validation loss

# import Python modules
import csv                      # csv reader
import matplotlib.pyplot as plt # matplotlib
import os                       # listdir
import sys                      # argv

# "constants"
EPC_IDX = 0 # index for epoch number
TRN_IDX = 1 # index for training loss
VAL_IDX = 2 # index for validation loss

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
   'python3 viz_trn_val.py /path/to/src/losses.csv /path/to/dst/'\
  )
  exit()

# read data from CSV file
csv_data = [[],[],[]]
with open(src, mode='r', newline='') as ifile:
  csvreader = csv.reader(ifile)
  header = next(csvreader)
  for row in csvreader:
    csv_data[EPC_IDX].append(float(row[0]))
    csv_data[TRN_IDX].append(float(row[1]))
    csv_data[VAL_IDX].append(float(row[2]))

# plot csv data
plt_title = 'Training and validation loss by epoch'
plt_y_axis = 'Loss'
plt_x_axis = 'Epoch'
fig = plt.figure(layout='constrained')
plt.plot(\
 csv_data[EPC_IDX], csv_data[TRN_IDX],\
 marker='.', linestyle='None', label='Training loss'
)
plt.plot(\
 csv_data[EPC_IDX], csv_data[VAL_IDX],\
 marker='.', linestyle='None', label='Validation loss'
)
plt.title(plt_title)
plt.ylabel(plt_y_axis)
plt.xlabel(plt_x_axis)
plt.legend()
plt.savefig(os.path.join(dst,'viz-loss.pdf'),format='pdf')
plt.close(fig)
