#!/bin/bash
#
# setup_p3_venv.sh
# A bash script to set up Python 3 virtual environment
# See https://pytorch.org/get-started/locally/#start-locally for updates
# For other modules, check available versions: python3 -m pip index versions [.]
#
# Usage: ./setup_p3_venv.sh
#  - Must execute from 01-setup directory
# Prerequisites:
#  - sudo apt install python3-tk
#  - sudo apt install python3-pip
#  - sudo apt install python3-venv
# Arguments:
#  - None
# Outputs:
#  - Python 3 virtual environment

cd ../
python3 -m venv p3-env
source p3-env/bin/activate
python3 -m pip install numpy==2.3.5
python3 -m pip install matplotlib==3.10.8
python3 -m pip install tqdm==4.67.1
python3 -m pip install torch torchvision \
 --index-url https://download.pytorch.org/whl/cpu
deactivate
