#!/bin/bash
######################## start of slurm options #########################################
########################################
# Job Identification & Runtime         #
########################################
#SBATCH --job-name=Smart_Energy_1024_6        # Job name
#SBATCH --account=personal                  # Account name
#SBATCH --time=45:00:00                     # Time limit (HH:MM:SS)
#SBATCH --partition=normal_q                # Partition name
#SBATCH --output=job_output_%j.out          # Standard output file (%j = job ID)
#SBATCH --error=job_error_%j.err            # Standard error file
#######################################
# CPU and Node Configuration          #
#######################################
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks (processes) per node
#SBATCH --cpus-per-task=64                   # Number of CPUs per task (threading)
#SBATCH --mem=64G                           # Memory per node
########################## end of slurm options #########################################

# Description: This is an example slurm batch job script to run seaborn in python
# Usage: To run this script, run the command 'sbatch example.slurm'

#########################################################################################
# Resets module system (recommended)
module reset


# Set up virtual environment
module load Miniforge3/25.11.0-1
eval "$(/apps/common/software/Miniforge3/25.11.0-1/bin/conda shell.bash hook)"
source activate smart-energy

echo "ENV=$CONDA_DEFAULT_ENV"
echo "PY=$(which python)"
python -c "import numpy; print('numpy ok')"

IDX = 6
NODE_IDX = 1024

# Run the python script
cd "$SLURM_SUBMIT_DIR"
cd 02-gen-data

python ./gen_data.py \
    ./test.csv \
    ../03-data/

echo "Generated Data!"

# Split Data
cd ../05-split-data

python ./split_data.py \
    ../03-data/ ./

echo "Split Data!"

# Run Training
cd ../06-trn-val-mlp/

python trn_val_mlp.py \
    ./mlp-${NODE_IDX}-cfg/mlp-${NODE_IDX}-${IDX}.json \
    ../05-split-data/ \
    ./mlp-${NODE_IDX}-results/mlp-${NODE_IDX}-${IDX}/

echo "Trained Model!"

# Vizualize Results
cd ../07-viz-results/

python viz_trn_val.py \
    ../06-trn-val-mlp/mlp-${NODE_IDX}-results/mlp-${NODE_IDX}-${IDX}/mlp-${NODE_IDX}-${IDX}-losses.csv \
    ../06-trn-val-mlp/mlp-${NODE_IDX}-results/mlp-${NODE_IDX}-${IDX}/

echo "Vizualized Loss!"

# Run Test
cd ../08-tst-mlp/

python tst_mlp.py \ 
    ../06-trn-val-mlp/mlp-${NODE_IDX}-cfg/mlp-${NODE_IDX}-${IDX}.json \
    ../06-trn-val-mlp/mlp-${NODE_IDX}-results/mlp-${NODE_IDX}-${IDX}/mlp-${NODE_IDX}-${IDX}.pt \
    ../06-trn-val-mlp/mlp-${NODE_IDX}-results/mlp-${NODE_IDX}-${IDX}/mlp-${NODE_IDX}-${IDX}-norm.pt \
    ../05-split-data/ \
    ./mlp-${NODE_IDX}-results/

echo "Deployed Model!"

# Deactivate virtual environment
conda deactivate