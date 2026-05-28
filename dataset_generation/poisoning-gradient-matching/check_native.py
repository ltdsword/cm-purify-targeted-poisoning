import torch
import forest

args_list = [
    '--dataset', 'CIFAR10',
    '--targets', '10',
    '--budget', '0.01',
    '--net', 'ResNet18',
    '--poisonkey', '3275641999',
    '--modelkey', '177672595',
    '--vruns', '1'
]
args = forest.options().parse_args(args_list)
setup = forest.utils.system_startup(args)

model = forest.Victim(args, setup=setup)
kettle = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
witch = forest.Witch(args, setup=setup)

print("Starting to train baseline clean model...")
# stats_clean = model.train(kettle, max_epoch=40)

print(f"Loading our poisoned dataset from 'poisons/train'...")
import torchvision.datasets as datasets
import torchvision.transforms as transforms
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
clean_train_dataset = datasets.ImageFolder('poisons/train', transform=transform_train)
# Wait, kettle needs `trainloader` to have these inputs. 
kettle.trainloader = torch.utils.data.DataLoader(clean_train_dataset, batch_size=128, shuffle=True, num_workers=4)

print("Training model...")
# Actually, their `test` expects `poison_delta` to be passed, but since the images ALREADY have the delta baked in when we saved them, we pass None!
# We can just call their train method.
_ = model.train(kettle, max_epoch=40)

print("Validating with their original code...")
# This will natively evaluate using their check_targets function
stats_results = model.validate(kettle, poison_delta=None)

print(f"Validation Target Success: {stats_results}")
