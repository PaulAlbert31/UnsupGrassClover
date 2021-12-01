import torchvision as tv
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from torch.utils.data import Dataset
import torch
import os
import csv
from tqdm import tqdm

class Irish(Dataset):
    # including hard labels & soft labels
    def __init__(self, data, labels, transform=None, pertu=False):
        self.train_data, self.train_labels =  data, labels
        self.transform = transform
        self.pertu = pertu
        
    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
            
        img = Image.open(img)
        img = self.transform(img)
        sample = {'image':img, 'target':target, 'index':index}
        return sample

    def __len__(self):
        return len(self.train_data)


def make_dataset(root, red=False):
    np.random.seed(42)
    img_paths = []
    labels = torch.tensor([])
    if red:
        files = [os.path.join(root, "train_red.csv"), os.path.join(root, "val.csv")]
    else:
        files = [os.path.join(root, "train_full.csv"), os.path.join(root, "val.csv")]
    for i, gt_file in enumerate(files):
        if i == 1:
            train_num = len(img_paths)
        with open(gt_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i < 1:
                    continue
                else:
                    if row[-1] == "missing data" or float(row[1]) == 0:
                        continue
                    img_paths.append(os.path.join(root, 'images', row[0].replace('JPG','jpg')))
                    biomass = torch.tensor([float(r)/100 for r in row[2:5]])
                    herbage_mass = torch.tensor([float(row[1])])
                    herbage_height = torch.tensor([float(row[-1])])
                    lab = torch.cat((herbage_mass, herbage_height, biomass), dim=0).view(1,-1)
                    labels = torch.cat((labels, lab))

    # split in train and validation:
    img_paths = np.asarray(img_paths)
    
    train_paths = img_paths[:train_num]
    train_labels = labels[:train_num]
    
    val_paths = img_paths[train_num:]
    val_labels = labels[train_num:]

    return train_paths, train_labels, val_paths, val_labels
