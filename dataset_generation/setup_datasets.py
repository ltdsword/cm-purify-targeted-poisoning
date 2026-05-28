import os
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np

def main():
    print("Setting up global datasets for Witches Brew and Bullseye Polytope...")
    os.makedirs('/media02/ndthuc03/cm-purify-targeted-poisoning/BullseyePoison/datasets', exist_ok=True)
    os.makedirs('/media02/ndthuc03/cm-purify-targeted-poisoning/dataset_generation/configs', exist_ok=True)
    
    # Download / Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='/media02/ndthuc03/cm-purify-targeted-poisoning/datasets', train=True, download=True, transform=None)
    testset = torchvision.datasets.CIFAR10(root='/media02/ndthuc03/cm-purify-targeted-poisoning/datasets', train=False, download=True, transform=None)

    # 1. Create CIFAR10_TRAIN_Split.pth for Bullseye Polytope
    train_list = [(img, label) for img, label in trainset]
    test_list = [(img, label) for img, label in testset]
    
    bp_data = {
        'clean_train': train_list,
        'others': train_list,
        'target': test_list
    }
    torch.save(bp_data, '/media02/ndthuc03/cm-purify-targeted-poisoning/BullseyePoison/datasets/CIFAR10_TRAIN_Split.pth')
    print("Saved BullseyePoison/datasets/CIFAR10_TRAIN_Split.pth")

    # 2. Build explicit Witches Brew Benchmark files
    np.random.seed(123)
    train_labels = np.array(trainset.targets)
    test_labels = np.array(testset.targets)

    # We will build benchmarks per class 0 to 9
    wb_configs = []
    
    for cls in range(10):
        # find all train indices belonging to this class
        cls_idx = np.where(train_labels == cls)[0]
        np.random.shuffle(cls_idx)

        # PLAN.md partitioning:
        # 0-499:   WB Train Base 1
        # 500-999: WB Train Base 2
        # 1000-1499: WB Eval Base

        wb_train_1 = cls_idx[0:500].tolist()
        wb_train_2 = cls_idx[500:1000].tolist()
        wb_eval = cls_idx[1000:1500].tolist()
        
        # Test target indexing (must be in test set per kettle validset)
        # Select target images from test set (just for WB)
        cls_test_idx = np.where(test_labels == cls)[0]
        np.random.shuffle(cls_test_idx)
        
        config_t1 = {
            "target index": cls_test_idx[0].item(), 
            "base indices": wb_train_1
        }
        config_t2 = {
            "target index": cls_test_idx[1].item(), 
            "base indices": wb_train_2
        }
        config_e = {
            "target index": cls_test_idx[2].item(), 
            "base indices": wb_eval
        }
        
        # Save to wb config dictionaries
        fn_t1 = f'/media02/ndthuc03/cm-purify-targeted-poisoning/dataset_generation/configs/wb_c{cls}_train1.pickle'
        with open(fn_t1, 'wb') as f: pickle.dump([config_t1], f)
            
        fn_t2 = f'/media02/ndthuc03/cm-purify-targeted-poisoning/dataset_generation/configs/wb_c{cls}_train2.pickle'
        with open(fn_t2, 'wb') as f: pickle.dump([config_t2], f)
            
        fn_e = f'/media02/ndthuc03/cm-purify-targeted-poisoning/dataset_generation/configs/wb_c{cls}_eval.pickle'
        with open(fn_e, 'wb') as f: pickle.dump([config_e], f)
            
    print("Saved WB Configs to /global/configs/")

if __name__ == "__main__":
    main()
