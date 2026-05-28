#!/bin/bash
#SBATCH --job-name=generate_poisons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=32G              
#SBATCH --gres=gpu:1           
#SBATCH --time=48:00:00        
#SBATCH --output=cm-purify-targeted-poisoning/dataset_generation/logs/%j_generate.log

set -e

# Load conda environment
eval "$(conda shell.bash hook)"
conda activate targeted_poisoning
pip install requirements.txt

# Run the single MAIN script to execute both Setup and Poison Generation
echo "=============================="
echo "1. PREPARING CLEAN DATASETS..."
echo "=============================="
python cm-purify-targeted-poisoning/dataset_generation/scripts/dataset_generation.py --mode setup_clean

echo "=============================="
echo "2. GENERATING WTICHES BREW POISONS..."
echo "=============================="
python cm-purify-targeted-poisoning/dataset_generation/scripts/dataset_generation.py --mode craft_wb

echo "=============================="
echo "3. GENERATING BULLSEYE POLYTOPE POISONS..."
echo "=============================="
python cm-purify-targeted-poisoning/dataset_generation/scripts/dataset_generation.py --mode craft_bp

echo "=============================="
echo "DONE! Results are in cm-purify-targeted-poisoning/dataset_generation/datasets/"
echo "=============================="
