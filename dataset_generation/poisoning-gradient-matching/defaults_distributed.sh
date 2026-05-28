#!/bin/bash
#SBATCH --job-name=defaults_dist
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=log/defaults_dist_%j.out

# Complying with server restrictions: 
# Max 48h runtime, Max 2 GPUs (max limit), 16 CPUs (max limit), 64G RAM (max limit).

# Activate required conda env
eval "$(conda shell.bash hook)"
conda activate targeted_poisoning

# ResNet default distributed
# Note: Reduced nproc_per_node from 4 to 2 to comply with max 2 GPUs server limit
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1000000000
python -m torch.distributed.launch --nproc_per_node=2 --master_port=28502 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1100000000
python -m torch.distributed.launch --nproc_per_node=2 --master_port=27503 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1110000000
python -m torch.distributed.launch --nproc_per_node=2 --master_port=26504 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111000000
python -m torch.distributed.launch --nproc_per_node=2 --master_port=25505 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111100000
python -m torch.distributed.launch --nproc_per_node=2 --master_port=24506 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111110000
python -m torch.distributed.launch --nproc_per_node=2 --master_port=23507 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111111000
python -m torch.distributed.launch --nproc_per_node=2 --master_port=22508 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111111100
python -m torch.distributed.launch --nproc_per_node=2 --master_port=21509 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111111110
python -m torch.distributed.launch --nproc_per_node=2 --master_port=20510 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111111111