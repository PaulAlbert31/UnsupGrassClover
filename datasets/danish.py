import torchvision as tv
import numpy as np
from PIL import Image
import time
from torch.utils.data import Dataset
import torch
import os
import csv
from tqdm import tqdm



class Danish(Dataset):
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


def make_dataset(root):
    np.random.seed(42)
    img_paths = []
    labels = torch.tensor([])
    files = [os.path.join(root, path) for path in ["train.csv", "val.csv"]]
    for j, gt_file in enumerate(files):
        with open(gt_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for i, row in enumerate(csv_reader):
                if i == 0:
                    print(row[-5:])
                else:
                    if row[3] != 'basic':
                        img_paths.append(os.path.join(root, 'images', row[0]))
                        imp = row[-5:]
                        labels = torch.cat((labels, torch.tensor([float(imp[3]), float(imp[0]), float(imp[2]), float(imp[1]), float(imp[4])]).unsqueeze(0))) #[3,0,2,1,4]
        if j == 0:
            train_num = len(img_paths)
            
    img_paths = np.array(img_paths)
    # split in train and validation:
    train_paths = img_paths[:train_num]
    train_labels = labels[:train_num]
    val_paths = img_paths[train_num:]
    val_labels = labels[train_num:]
    

    return train_paths, train_labels, val_paths, val_labels
