import os
import pickle
import numpy as np
import torchvision
import random
import argparse
import subprocess
import sys
from PIL import Image

# Directory configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(BASE_DIR)
DATA_ROOT = os.path.join(BASE_DIR, "datasets")
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
SLURM_DIR = os.path.join(BASE_DIR, "slurm_jobs")
BP_ROOT = os.path.join(BASE_DIR, "BullseyePoison")
BP_DATA_DIR = os.path.join(BP_ROOT, "datasets")
BP_SPLIT_PATH = os.path.join(BP_DATA_DIR, "CIFAR10_TRAIN_Split.pth")

# Final destination folders where paired datasets will be saved
TRAIN_CLEAN_DIR = os.path.join(DATA_ROOT, "train", "clean")
TRAIN_POISON_DIR = os.path.join(DATA_ROOT, "train", "poisons")
TEST_DIR = os.path.join(DATA_ROOT, "test")

SEED = 121 
BP_MODE = os.environ.get("BP_MODE", "mean")
BP_POISON_ITERS = os.environ.get("BP_POISON_ITERS", "1500")

def _all_files_exist(paths):
    return all(os.path.exists(path) for path in paths)

def latest_checkpoint_iteration(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None

    latest = None
    for name in os.listdir(checkpoint_dir):
        if not name.startswith("poison_") or not name.endswith(".pth"):
            continue
        try:
            iteration = int(name[len("poison_"):-len(".pth")])
        except ValueError:
            continue
        if latest is None or iteration > latest:
            latest = iteration
    return latest

def clean_train_name(class_idx, image_idx):
    return f"clean_c{class_idx}_{image_idx}.png"

def wb_name(class_idx, image_idx):
    return f"wb_c{class_idx}_{image_idx}.png"

def bp_name(class_idx, group_idx, image_idx):
    return f"bp_c{class_idx}_g{group_idx}_{image_idx}.png"

def target_name(class_idx, image_idx):
    return f"target_c{class_idx}_{image_idx}.png"

def wb_eval_dir(class_idx):
    return os.path.join(TEST_DIR, f"WB_c{class_idx}")

def bp_eval_dir(class_idx, group_idx):
    return os.path.join(TEST_DIR, f"BP_c{class_idx}_g{group_idx}")

def save_bp_dataset_split(train_set, test_set, shuffled_train_indices):
    """Create the BP split in the class-relative order used by PLAN.md.

    The official BP datasets.zip split only has 200 CIFAR images per class, so it
    cannot support our [1500..2499] BP train pool. BP only requires that
    fetch_target/fetch_poison_bases can iterate images by label, so we store a
    compact full-CIFAR split and teach BullseyePoison/utils.py to read it.
    """
    os.makedirs(BP_DATA_DIR, exist_ok=True)
    ordered_indices = np.concatenate([shuffled_train_indices[c] for c in range(10)])
    ordered_targets = np.array(train_set.targets, dtype=np.int64)[ordered_indices]
    split = {
        'format': 'cm_custom_compact_v1',
        'clean_train': {
            'data': train_set.data[ordered_indices],
            'targets': ordered_targets,
        },
        'others': {
            'data': train_set.data[ordered_indices],
            'targets': ordered_targets,
        },
        'target': {
            'data': test_set.data,
            'targets': np.array(test_set.targets, dtype=np.int64),
        },
    }
    with open(BP_SPLIT_PATH, 'wb') as f:
        pickle.dump(split, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved full BullseyePoison CIFAR split to {BP_SPLIT_PATH}")

    return {
        c: train_set.data[shuffled_train_indices[c]]
        for c in range(10)
    }

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
    os.makedirs(BP_DATA_DIR, exist_ok=True)

    print(f"Downloading/Verifying CIFAR-10 datasets in {DATA_ROOT}...")
    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    shuffled_train_indices = {}
    for c in range(10):
        class_indices = np.where(train_labels == c)[0]
        assert len(class_indices) == 5000
        np.random.shuffle(class_indices)
        shuffled_train_indices[c] = class_indices

    bp_class_data = save_bp_dataset_split(train_set, test_set, shuffled_train_indices)

    wb_setups = []
    bp_setups = []

    def pull_bp_target(base_class):
        target_classes = [cls for cls in range(10) if cls != base_class]
        target_cls = int(np.random.choice(target_classes))
        target_arg_idx = int(np.random.randint(0, 50))
        # craft_poisons_transfer.py calls fetch_target(..., start_idx=50).
        target_img = Image.fromarray(bp_class_data[target_cls][target_arg_idx + 50])
        return target_arg_idx, target_cls, target_img

    for c in range(10):
        class_indices = shuffled_train_indices[c]
        
        # Witches’ Brew Bounds
        wb_train1_pool = class_indices[0:500]       
        wb_train2_pool = class_indices[500:1000]    
        wb_eval_pool   = class_indices[1000:1500]   
        
        # Bullseye Polytope Bounds
        bp_train_pool  = list(range(1500, 2500))
        bp_eval_pool   = list(range(2500, 2520))
        
        # Clean Dataset
        clean_cm_pool  = class_indices[2520:3520]   
        
        # Save Clean CM directly
        for idx in clean_cm_pool:
            img, _ = train_set[idx]
            img.save(os.path.join(TRAIN_CLEAN_DIR, clean_train_name(c, idx)))
        
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
                'target class': t_cls, 'base class': c, 'desc': suffix, 'is_train': is_train
            }
            wb_setups.append(struct)
            
            # Save clean bases to TRAIN cleanly
            for b_idx in pool:
                img, _ = train_set[b_idx]
                dest = TRAIN_CLEAN_DIR if is_train else os.path.join(wb_eval_dir(c), "clean")
                if not is_train: os.makedirs(dest, exist_ok=True)
                img.save(os.path.join(dest, wb_name(c, b_idx)))
                
            # If Eval, save the Target image directly
            if not is_train:
                t_dest = os.path.join(wb_eval_dir(c), "target")
                os.makedirs(t_dest, exist_ok=True)
                t_img.save(os.path.join(t_dest, target_name(t_cls, t_idx)))

        # ------------------------------------------
        # BP Setup 
        # ------------------------------------------
        # Train (100 distinct groups of 10)
        for g in range(100):
            t_idx, t_cls, t_img = pull_bp_target(c)
            sub_indices = bp_train_pool[g*10 : (g+1)*10]
            bp_setups.append({
                'base indices': list(sub_indices), 'target index': t_idx,
                'target class': t_cls, 'base class': c, 'desc': 'BP_Train', 
                'batch_group': g, 'is_train': True, 'start_idx': 1500 + (g*10)
            })
            for b_idx in sub_indices:
                img = Image.fromarray(bp_class_data[c][b_idx])
                img.save(os.path.join(TRAIN_CLEAN_DIR, bp_name(c, g, b_idx)))
                
        # Eval (2 distinct groups of 10)
        for g in range(2):
            t_idx, t_cls, t_img = pull_bp_target(c)
            sub_indices = bp_eval_pool[g*10 : (g+1)*10]
            bp_setups.append({
                'base indices': list(sub_indices), 'target index': t_idx,
                'target class': t_cls, 'base class': c, 'desc': 'BP_Eval', 
                'batch_group': g, 'is_train': False, 'start_idx': 2500 + (g*10)
            })
            dest = bp_eval_dir(c, g)
            os.makedirs(os.path.join(dest, "clean"), exist_ok=True)
            t_dest = os.path.join(dest, "target")
            os.makedirs(t_dest, exist_ok=True)
            t_img.save(os.path.join(t_dest, target_name(t_cls, t_idx)))
            for b_idx in sub_indices:
                img = Image.fromarray(bp_class_data[c][b_idx])
                img.save(os.path.join(dest, "clean", bp_name(c, g, b_idx)))

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

        if setup['is_train']:
            expected_outputs = [
                os.path.join(TRAIN_POISON_DIR, wb_name(setup['base class'], base_idx))
                for base_idx in setup['base indices']
            ]
        else:
            dest = os.path.join(wb_eval_dir(setup['base class']), "poisons")
            expected_outputs = [
                os.path.join(dest, wb_name(setup['base class'], base_idx))
                for base_idx in setup['base indices']
            ]

        if _all_files_exist(expected_outputs):
            print(f"WB {i+1}/{len(wb_setups)} already exported; skipping.")
            continue

        result_dir = os.path.join(wb_root, 'poisons', 'benchmark_results', f"wb_{i}_ResNet18", str(i))
        poisons_file = os.path.join(result_dir, 'poisons.pickle')

        if os.path.exists(poisons_file):
            print(f"Found existing WB result at {poisons_file}; exporting PNGs.")
        else:
            cmd = [
                sys.executable, "brew_poison.py",
                "--name", f"wb_{i}",
                "--benchmark", os.path.join(CONFIG_DIR, 'wb_benchmark_setups.pickle'),
                "--save", "benchmark", "--vruns", "0", "--eps", "8",
                "--benchmark_idx", str(i), "--ensemble", "1", "--net", "ResNet18"
            ]

            subprocess.run(cmd, cwd=wb_root, check=True)
        
        if os.path.exists(poisons_file):
            with open(poisons_file, 'rb') as pf:
                poison_data = pickle.load(pf) 
                
            for p_num, (p_img, p_label) in enumerate(poison_data):
                base_idx = setup['base indices'][p_num]
                if setup['is_train']:
                    p_img.save(os.path.join(TRAIN_POISON_DIR, wb_name(setup['base class'], base_idx)))
                else: 
                    dest = os.path.join(wb_eval_dir(setup['base class']), "poisons")
                    os.makedirs(dest, exist_ok=True)
                    p_img.save(os.path.join(dest, wb_name(setup['base class'], base_idx)))
        else:
            raise FileNotFoundError(f"Expected WB poisons were not found at {poisons_file}")


def craft_bp():
    with open(os.path.join(CONFIG_DIR, 'bp_benchmark_setups.pickle'), 'rb') as f:
        bp_setups = pickle.load(f)
        
    bp_root = os.path.join(BASE_DIR, 'BullseyePoison')
    os.makedirs(TRAIN_POISON_DIR, exist_ok=True)
    
    # Ensure model-chks exists for BullseyePoison. The downloaded zip may
    # extract as model-chks-release, but craft_poisons_transfer.py loads
    # from model-chks by default.
    bp_model_chks = os.path.join(bp_root, 'model-chks')
    bp_model_chks_alias = os.path.join(bp_root, 'model-chks-release')

    def ensure_checkpoint_alias():
        if os.path.exists(bp_model_chks_alias):
            return
        try:
            os.symlink('model-chks', bp_model_chks_alias)
        except OSError:
            pass

    def has_checkpoints(path):
        return os.path.isdir(path) and bool(os.listdir(path))

    def populate_bp_checkpoints():
        import shutil
        extracted_dirs = [
            os.path.join(bp_root, 'model-chks-release'),
            os.path.join(bp_root, 'model_chks_release'),
        ]
        os.makedirs(bp_model_chks, exist_ok=True)
        for extracted_dir in extracted_dirs:
            if extracted_dir == bp_model_chks:
                continue
            if os.path.islink(extracted_dir):
                continue
            if not os.path.isdir(extracted_dir):
                continue
            for file_name in os.listdir(extracted_dir):
                shutil.move(os.path.join(extracted_dir, file_name), bp_model_chks)
            shutil.rmtree(extracted_dir)
        ensure_checkpoint_alias()
        return os.listdir(bp_model_chks)

    if not has_checkpoints(bp_model_chks):
        if populate_bp_checkpoints():
            print("Moved extracted BullseyePoison checkpoints into model-chks.")
        else:
            print("model-chks missing or empty for BullseyePoison. Downloading from Google Drive...")
            try:
                import gdown
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
                import gdown
                
            zip_path = os.path.join(bp_root, "model_chks_release.zip")
            gdown.download(id="1TwxNbJ1arDNQrBJdt5AFeaAbKC65HOko", output=zip_path, quiet=False)
            
            print("Extracting checkpoints...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(bp_root)
            populate_bp_checkpoints()
            os.remove(zip_path)
        print("Checkpoints downloaded and extracted successfully.")
    else:
        ensure_checkpoint_alias()

    print(f"Total BP setups: {len(bp_setups)}")
    for i, setup in enumerate(bp_setups):
        print(f"Crafting BP {i+1}/{len(bp_setups)} - Class {setup['base class']} Group {setup['batch_group']}")

        if setup['is_train']:
            expected_outputs = [
                os.path.join(TRAIN_POISON_DIR, bp_name(setup['base class'], setup['batch_group'], base_idx))
                for base_idx in setup['base indices']
            ]
        else:
            dest = os.path.join(bp_eval_dir(setup['base class'], setup['batch_group']), "poisons")
            expected_outputs = [
                os.path.join(dest, bp_name(setup['base class'], setup['batch_group'], base_idx))
                for base_idx in setup['base indices']
            ]

        if _all_files_exist(expected_outputs):
            print(f"BP {i+1}/{len(bp_setups)} already exported; skipping.")
            continue
        
        export_dir = os.path.join(bp_root, "benchmark_results", f"bp_{i}")
        env = os.environ.copy()
        env['BP_EXPORT_DIR'] = export_dir
        env['PYTHONPATH'] = REPO_DIR + os.pathsep + env.get('PYTHONPATH', '')
        poisons_file = os.path.join(export_dir, 'poisons.pickle')

        if os.path.exists(poisons_file):
            print(f"Found existing BP result at {poisons_file}; exporting PNGs.")
        else:
            checkpoint_dir = os.path.join(export_dir, BP_MODE, BP_POISON_ITERS, str(setup['target index']))
            resume_ite = latest_checkpoint_iteration(checkpoint_dir)
            cmd = [
                sys.executable, "craft_poisons_transfer.py",
                "--target-label", str(setup['target class']),
                "--target-index", str(setup['target index']),
                "--poison-label", str(setup['base class']),
                "--start-idx", str(setup['start_idx']),
                "--poison-num", "10",
                "--substitute-nets", "ResNet18",
                "--target-net", "ResNet18",
                "--model-resume-path", "model-chks",
                "--mode", BP_MODE,
                "--poison-ites", BP_POISON_ITERS,
            ]

            if resume_ite is not None and resume_ite > 0:
                print(
                    f"Found partial BP checkpoint for case {i} at "
                    f"{os.path.join(checkpoint_dir, f'poison_{resume_ite:05d}.pth')}; resuming."
                )
                cmd.extend(["--resume-poison-ite", str(resume_ite)])

            subprocess.run(cmd, cwd=bp_root, env=env, check=True)

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
                    p_img.save(os.path.join(TRAIN_POISON_DIR, bp_name(setup['base class'], setup['batch_group'], base_idx)))
                else:
                    dest = os.path.join(bp_eval_dir(setup['base class'], setup['batch_group']), "poisons")
                    os.makedirs(dest, exist_ok=True)
                    p_img.save(os.path.join(dest, bp_name(setup['base class'], setup['batch_group'], base_idx)))
        else:
            raise FileNotFoundError(f"Expected BP poisons were not found at {poisons_file}")

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
