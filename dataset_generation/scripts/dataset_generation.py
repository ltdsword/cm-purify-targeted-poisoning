import os
import pickle
import numpy as np
import torchvision
import random
import argparse
import subprocess
from PIL import Image

# Directory configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "datasets")
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
SLURM_DIR = os.path.join(BASE_DIR, "slurm_jobs")

# Final destination folders where paired datasets will be saved
TRAIN_CLEAN_DIR = os.path.join(DATA_ROOT, "train", "clean")
TRAIN_POISON_DIR = os.path.join(DATA_ROOT, "train", "poisons")
TEST_DIR = os.path.join(DATA_ROOT, "test")

SEED = 121 

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def setup_clean_datasets():
    """
    Splits CIFAR-10 exactly as PLAN.md requires and saves the x_clean 
    baseline images to the disk matching the pairs.
    """
    setup_seed(SEED)
    
    os.makedirs(TRAIN_CLEAN_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    print(f"Downloading/Verifying CIFAR-10 datasets in {DATA_ROOT}...")
    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    wb_setups = []
    bp_setups = []

    for c in range(10):
        class_indices = np.where(train_labels == c)[0]
        assert len(class_indices) == 5000
        
        # Shuffle immediately
        np.random.shuffle(class_indices)
        
        # Witches’ Brew Bounds
        wb_train1_pool = class_indices[0:500]       
        wb_train2_pool = class_indices[500:1000]    
        wb_eval_pool   = class_indices[1000:1500]   
        
        # Bullseye Polytope Bounds
        bp_train_pool  = class_indices[1500:2500]   
        bp_eval_pool   = class_indices[2500:2520]   
        
        # Clean Dataset
        clean_cm_pool  = class_indices[2520:3520]   
        
        # Save Clean CM directly
        for idx in clean_cm_pool:
            img, _ = train_set[idx]
            img.save(os.path.join(TRAIN_CLEAN_DIR, f"clean_baselines_c{c}_{idx}.png"))
        
        # Fetch unique target helper
        valid_test_indices = np.where(test_labels != c)[0]
        np.random.shuffle(valid_test_indices)
        t_ptr = 0
        def pull_target():
            nonlocal t_ptr
            target_id = valid_test_indices[t_ptr]
            target_cls = test_labels[target_id]
            t_ptr += 1
            return int(target_id), int(target_cls), test_set[target_id][0]

        # ------------------------------------------
        # WB Setup 
        # ------------------------------------------
        for suffix, pool, is_train in [('train1', wb_train1_pool, True), 
                                       ('train2', wb_train2_pool, True), 
                                       ('eval', wb_eval_pool, False)]:
            t_idx, t_cls, t_img = pull_target()
            struct = {
                'base indices': pool.tolist(), 'target index': t_idx, 
                'target class': t_cls, 'base class': c, 'desc': f'WB_{suffix}', 'is_train': is_train
            }
            wb_setups.append(struct)
            
            # Save clean bases to TRAIN cleanly
            for b_idx in pool:
                img, _ = train_set[b_idx]
                dest = TRAIN_CLEAN_DIR if is_train else os.path.join(TEST_DIR, f"WB_c{c}_{suffix}", "clean")
                if not is_train: os.makedirs(dest, exist_ok=True)
                img.save(os.path.join(dest, f"wb_c{c}_{suffix}_base_{b_idx}.png"))
                
            # If Eval, save the Target image directly
            if not is_train:
                t_dest = os.path.join(TEST_DIR, f"WB_c{c}_{suffix}", "target")
                os.makedirs(t_dest, exist_ok=True)
                t_img.save(os.path.join(t_dest, f"target_c{t_cls}_id{t_idx}.png"))

        # ------------------------------------------
        # BP Setup 
        # ------------------------------------------
        # Train (100 distinct groups of 10)
        for g in range(100):
            t_idx, t_cls, t_img = pull_target()
            sub_indices = bp_train_pool[g*10 : (g+1)*10]
            bp_setups.append({
                'base indices': sub_indices.tolist(), 'target index': t_idx, 
                'target class': t_cls, 'base class': c, 'desc': 'BP_Train', 
                'batch_group': g, 'is_train': True, 'start_idx': 1500 + (g*10)
            })
            for b_idx in sub_indices:
                img, _ = train_set[b_idx]
                img.save(os.path.join(TRAIN_CLEAN_DIR, f"bp_train_c{c}_g{g}_base_{b_idx}.png"))
                
        # Eval (2 distinct groups of 10)
        for g in range(2):
            t_idx, t_cls, t_img = pull_target()
            sub_indices = bp_eval_pool[g*10 : (g+1)*10]
            bp_setups.append({
                'base indices': sub_indices.tolist(), 'target index': t_idx, 
                'target class': t_cls, 'base class': c, 'desc': 'BP_Eval', 
                'batch_group': g, 'is_train': False, 'start_idx': 2500 + (g*10)
            })
            dest = os.path.join(TEST_DIR, f"BP_c{c}_eval_g{g}")
            os.makedirs(os.path.join(dest, "clean"), exist_ok=True)
            t_dest = os.path.join(dest, "target")
            os.makedirs(t_dest, exist_ok=True)
            t_img.save(os.path.join(t_dest, f"target_c{t_cls}_id{t_idx}.png"))
            for b_idx in sub_indices:
                img, _ = train_set[b_idx]
                img.save(os.path.join(dest, "clean", f"bp_eval_c{c}_g{g}_base_{b_idx}.png"))

    with open(os.path.join(CONFIG_DIR, 'wb_benchmark_setups.pickle'), 'wb') as f:
        pickle.dump(wb_setups, f)
    with open(os.path.join(CONFIG_DIR, 'bp_benchmark_setups.pickle'), 'wb') as f:
        pickle.dump(bp_setups, f)
        
    print("Clean images explicitly verified and saved!")


def craft_wb():
    with open(os.path.join(CONFIG_DIR, 'wb_benchmark_setups.pickle'), 'rb') as f:
        wb_setups = pickle.load(f)
        
    wb_root = os.path.join(BASE_DIR, 'poisoning-gradient-matching')
    os.makedirs(TRAIN_POISON_DIR, exist_ok=True)

    print(f"Total WB setups: {len(wb_setups)}")
    for i, setup in enumerate(wb_setups):
        print(f"Crafting WB {i+1}/{len(wb_setups)} - Class {setup['base class']} {setup['desc']}")
        
        cmd = [
            "python", "brew_poison.py", 
            "--name", f"wb_{i}", 
            "--benchmark", os.path.join(CONFIG_DIR, 'wb_benchmark_setups.pickle'), 
            "--save", "benchmark", "--vruns", "0", "--eps", "8", 
            "--benchmark_idx", str(i), "--ensemble", "1", "--net", "ResNet18"
        ]
        
        subprocess.run(cmd, cwd=wb_root, check=True)
        
        result_dir = os.path.join(wb_root, 'benchmark_results', f"wb_{i}_ResNet18", str(i))
        poisons_file = os.path.join(result_dir, 'poisons.pickle')
        
        if os.path.exists(poisons_file):
            with open(poisons_file, 'rb') as pf:
                poison_data = pickle.load(pf) 
                
            for p_num, (p_img, p_label) in enumerate(poison_data):
                base_idx = setup['base indices'][p_num]
                if setup['is_train']:
                    p_img.save(os.path.join(TRAIN_POISON_DIR, f"wb_c{setup['base class']}_{setup['desc']}_base_{base_idx}.png"))
                else: 
                    dest = os.path.join(TEST_DIR, f"WB_c{setup['base class']}_{setup['desc']}", "poisons")
                    os.makedirs(dest, exist_ok=True)
                    p_img.save(os.path.join(dest, f"wb_c{setup['base class']}_{setup['desc']}_base_{base_idx}.png"))


def craft_bp():
    with open(os.path.join(CONFIG_DIR, 'bp_benchmark_setups.pickle'), 'rb') as f:
        bp_setups = pickle.load(f)
        
    bp_root = os.path.join(BASE_DIR, 'BullseyePoison')
    os.makedirs(TRAIN_POISON_DIR, exist_ok=True)

    print(f"Total BP setups: {len(bp_setups)}")
    for i, setup in enumerate(bp_setups):
        print(f"Crafting BP {i+1}/{len(bp_setups)} - Class {setup['base class']} Group {setup['batch_group']}")
        
        export_dir = os.path.join(bp_root, "benchmark_results", f"bp_{i}")
        env = os.environ.copy()
        env['BP_EXPORT_DIR'] = export_dir
        
        cmd = [
            "python", "craft_poisons_transfer.py", 
            "--target-label", str(setup['target class']),
            "--target-index", str(setup['target index']),
            "--poison-label", str(setup['base class']),
            "--start-idx", str(setup['start_idx']),
            "--poison-num", "10",
            "--substitute-nets", "ResNet18",
            "--target-net", "ResNet18"
        ]
        
        subprocess.run(cmd, cwd=bp_root, env=env, check=True)
        
        poisons_file = os.path.join(export_dir, 'poisons.pickle')
        if os.path.exists(poisons_file):
            import torch
            with open(poisons_file, 'rb') as pf:
                bp_data = pickle.load(pf)
                
            for p_num, (p_tensor, _) in enumerate(bp_data['poisons']):
                if isinstance(p_tensor, torch.Tensor):
                    
                    # BP uses means/(std+eps). Let's un-normalize if necessary, but generally we can save tensors natively or denorm
                    # If BP exported already denormalized tensors, we can safely ToPILImage... Let's just do a direct save
                    p_img = torchvision.transforms.ToPILImage()(p_tensor.cpu().clamp(0, 1))
                else: 
                    p_img = p_tensor
                    
                base_idx = setup['base indices'][p_num] 
                if setup['is_train']:
                    p_img.save(os.path.join(TRAIN_POISON_DIR, f"bp_train_c{setup['base class']}_g{setup['batch_group']}_base_{base_idx}.png"))
                else:
                    dest = os.path.join(TEST_DIR, f"BP_c{setup['base class']}_eval_g{setup['batch_group']}", "poisons")
                    os.makedirs(dest, exist_ok=True)
                    p_img.save(os.path.join(dest, f"bp_eval_c{setup['base class']}_g{setup['batch_group']}_base_{base_idx}.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['setup_clean', 'craft_wb', 'craft_bp'])
    args = parser.parse_args()
    
    if args.mode == 'setup_clean':
        setup_clean_datasets()
    elif args.mode == 'craft_wb':
        craft_wb()
    elif args.mode == 'craft_bp':
        craft_bp()
