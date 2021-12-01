import torch
import torchvision.transforms as transforms
import datasets
import numpy as np

def make_data_loader(args, no_aug=False, transform=None, **kwargs):   
    if 'irish' in args.dataset:
        mean = (0.41637952, 0.5502375,  0.2436111) 
        std = (0.190736, 0.21874362, 0.15318967)
    elif 'danish' in args.dataset:
        mean = (0.3137, 0.4320, 0.1619)
        std = (0.1614, 0.1905, 0.1325)
    
    size = args.size

    if args.base_da: #Weak data augmentation
        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]) 
    else: #Stronger data augmentation
        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    #Labeled images of the Irish dataset
    if args.dataset == "irish":
        trainset, testset = datasets.irish(red=args.red, transform=transform_train, transform_test=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
    #Labeled images from the GrassClover dataset
    elif args.dataset == "danish":
        trainset, testset = datasets.danish(transform=transform_train, transform_test=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        raise NotImplementedError("Dataset {} in not implemented".format(args.dataset))
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader
