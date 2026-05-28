import os
import sys
import logging
import argparse

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import forest
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

args_list = [
    '--dataset', 'CIFAR10',
    '--targets', '10',
    '--budget', '0.01',
    '--net', 'ResNet18',
    '--poisonkey', '3275641999', 
    '--modelkey', '177672595',   
    '--vruns', '1',              
]

args = forest.options().parse_args(args_list)
setup = forest.utils.system_startup(args)

model = forest.Victim(args, setup=setup)
model.initialize(seed=args.modelkey)

kettle = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

clean_train_dataset = datasets.ImageFolder('poisons/train', transform=transform_train)
train_loader = DataLoader(clean_train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

kettle.trainloader = train_loader
print("Starting train!")
model.train(kettle, max_epoch=1)
print("Finished train!")
