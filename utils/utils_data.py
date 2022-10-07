# code for data prepare
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import torch
import random
import pdb

class VisiualLoader:
    def __init__(self, loader):
        self.loader = iter(loader)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            images, labels = next(self.loader)
            return images, labels
        except StopIteration as e:
            raise StopIteration


def get_loaders(args):
    kwargs = {'num_workers': args.num_workers,
              'batch_size': args.batch_size,
              'shuffle': False,
              'pin_memory': False}
    if not args.add_gaussian:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.045) #https://arxiv.org/pdf/1901.09981.pdf
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(root= os.path.join(args.data_root, args.dataset), train=True,
                                transform=transform_train,
                                download=True)
    testset = datasets.CIFAR10(root= os.path.join(args.data_root, args.dataset), train=False,
                                transform=transform_test,
                                download=True)

    trainloader = DataLoader(trainset, **kwargs)
    if args.dataset == "cifar10":
        test_data_size = 10000
    else:
        test_data_size = 10000
    testloader = DataLoader(testset, num_workers=0, batch_size= test_data_size,  shuffle=False, pin_memory=False)
    return trainloader, testloader


def get_testloader(args, train=False, subset_idx=None):
    kwargs = {'num_workers': 0,
              'batch_size':100,
              'shuffle': False,
              'pin_memory': True}
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if subset_idx is not None:
        testset = Subset(datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset), train=train,
                                transform=transform_test,
                                download=False), subset_idx)

    else:
        testset = datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset), train=train,
                                transform=transform_test,
                                download=False).to(args.device)
    testloader = DataLoader(testset, **kwargs)
    return testloader

class DistillationLoader:
    def __init__(self, seed, target):
        self.seed = iter(seed)
        self.target = iter(target)

    def __len__(self):
        return len(self.seed)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            si, sl = next(self.seed)
            ti, tl = next(self.target)
            return si, sl, ti, tl
        except StopIteration as e:
            raise StopIteration


# This is used for training of GAL
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., prob=.5):
        self.std = std
        self.mean = mean
        self.prob = prob
        
    def __call__(self, tensor):
        if random.random() > self.prob:
            return tensor
        else:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
