# Multi-layer perceptron for predicting function families

MLP for predicting families of functions

## Directory Contents

* [01-setup](01-setup/README.md): Setup scripts
* [02-gen-data](01-download/README.md): Generate a dataset
* [03-data](03-data/README.md): A directory for holding generated data
* [04-viz-data](04-viz-data/README.md): Visualize a dataset
* [05-split-data](05-split-data/README.md): Splits dataset into training,
  validation, and testing sets
* [06-trn-val-mlp](06-trn-val-mlp/README.md): MLP training and validation
* [07-viz-trn-val](07-viz-trn-val/README.md): Visualize training and validation
* [08-tst-mlp](08-tst-mlp/README.md): Performs MLP inference
* [09-viz-mlp](09-viz-mlp/README.md): Visualizes MLP inference
* [README.md](README.md): This document

## Quickstart

Clone the repository and change directories into the repository.

```bash
cd ~
mkdir -p git-repos
cd ./git-repos/
git clone git@github.com:starbelt/repo-name.git
cd repo-name/
git config --global user.name "First Last"
git config --global user.email "flast@vt.edu"
```

Ensure prerequisites are installed.

```bash
sudo apt update
sudo apt upgrade
sudo apt install python3-tk
sudo apt install python3-pip
sudo apt install python3-venv
```

Run the setup script.

```bash
cd 01-setup/
./setup_p3_venv.sh
```

Activate the virtual environment.

```bash
cd ../
source p3-env/bin/activate
```

Generate the dataset.

```bash
cd ./02-gen-data/
python3 gen_data.py ./data-cfg.json ../03-data/
```

Visualize the dataset.

```bash
cd ../
cd ./04-viz-data/
python3 viz_data.py ../03-data/ ./
```

Split the dataset for training and evaluation.

```bash
cd ../
cd ./05-split-data/
python3 split_data.py ../03-data/ ./
```

Train and validate MLP models.

```bash
cd ../
cd ./06-trn-val-mlp/
python3 trn_val_mlp.py ./mlp-0256-0256-0256.json ../05-split-data/ ./
```

Visualize training and validation results.

```bash
cd ../
cd ./07-viz-trn-val/
python3 viz_trn_val.py ../06-trn-val-mlp/mlp-0256-0256-0256-losses.csv ./
```

Deploy a trained model.

```bash
cd ../
cd ./08-tst-mlp/
python3 tst_mlp.py ../06-trn-val-mlp/mlp-0256-0256-0256.json ../06-trn-val-mlp/mlp-0256-0256-0256.pt ../05-split-data/ ./
```

Visualize results.

```bash
cd ../
cd ./09-viz-mlp/
python3 viz_mlp.py ../06-trn-val-mlp/mlp-0256-0256-0256.json ../06-trn-val-mlp/mlp-0256-0256-0256.pt ../03-data/ ./
```

Deactivate virtual environment.

```bash
cd ../
deactivate
```

