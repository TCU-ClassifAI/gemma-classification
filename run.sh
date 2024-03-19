#!/bin/bash

# CD to the directory of the script
cd /home/classgpu/gemma-classification-1

# Ensure that the virtual environment is activated
source /home/classgpu/gemma-classification-1/venv-3.10/bin/activate

# Feel free to remake the venv using requirements.txt. There shouldn't be any conflicts.

# Run the script
python /home/classgpu/gemma-classification-1/classify_gemma.py
