import torchvision as tv
import numpy as np
from PIL import Image
import time
from torch.utils.data import Dataset
import torch
import os
import csv
from tqdm import tqdm



class Danish_Unsup(Dataset):
    # including hard labels & soft labels
    def __init__(self, data, labels, train=True, transform=None):
        self.train_data, self.train_labels =  data, labels
        self.train = train
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
            
        img = Image.open(img)
        
        if self.train:
            return {'image1':self.transform(img), 'image2': self.transform(img), 'target':target, 'index':index}

        return {'image':self.transform(img), 'target': target, 'index':index}



    def __len__(self):
        return len(self.train_data)


def make_dataset_clover(root):
    np.random.seed(42)
    img_paths = []
    for files in os.listdir(root):
        img_paths.append(os.path.join(root, files))

    # split in train and validation:
    idxes = np.random.permutation(len(img_paths))
    img_paths = np.asarray(img_paths)[idxes]
    no_label = torch.zeros(len(img_paths))

    return img_paths, no_label, None, None
