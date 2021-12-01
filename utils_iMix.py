import torch
import torchvision.transforms as transforms
import datasets
import numpy as np
from PIL import Image


def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    device = x.get_device()
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def make_data_loader(args, no_aug=False, transform=None, **kwargs):
    
    if args.dataset == 'irish':
        mean = (0.35271966, 0.46308401, 0.2102307)
        std = (0.23476366, 0.28419253, 0.17398185)
    elif args.dataset == 'danish':
        mean = (0.3137, 0.4320, 0.1619)
        std = (0.1614, 0.1905, 0.1325)


    sizer = 256
    size = 224

    transform_train = transforms.Compose([
        transforms.Resize(sizer),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    if args.dataset == "irish":
        trainset, _ = datasets.irish_iMix(transform=transform_train)
    elif args.dataset == "danish":
        trainset, _ = datasets.danish_iMix(transform=transform_train)
    else:
        raise NotImplementedError("Dataset {} in not implemented".format(args.dataset))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = None
    return train_loader, test_loader

