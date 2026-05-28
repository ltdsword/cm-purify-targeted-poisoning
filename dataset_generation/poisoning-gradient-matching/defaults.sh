#!/bin/bash
#SBATCH --job-name=defaults
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=log/defaults_%j.out

# Complying with server restrictions: 
# Max 48h runtime, Max 1 GPU (below 2 limit), 8 CPUs (below 16 limit), 32G RAM (below 64G limit).

# Activate required conda env
eval "$(conda shell.bash hook)"
conda activate targeted_poisoning

# ResNet default
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2000000000
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2100000000
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2110000000
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2111000000
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2111100000
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2111110000
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2111111000
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2111111100
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2111111110
python brew_poison.py  --net ResNet18 --vruns 8 --name def --poisonkey 2111111111
