import torchvision as tv
import numpy as np
from PIL import Image
import time
from torch.utils.data import Dataset
import torch
import os
import csv
from tqdm import tqdm


class Clover_Unsup(Dataset):
    def __init__(self, data, train=True, transform=None):
        self.train_data =  data
        self.train = train
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.train_data[index], 0
        img = Image.open(img)
        #img.draft('RGB',(img.size[0]//2, img.size[1]//2))
        #img = tv.io.read_image(img)
        if self.train:
            return {'image1':self.transform(img), 'image2': self.transform(img), 'target':target, 'index':index}

        return {'image':self.transform(img), 'target': target, 'index':index}

    def __len__(self):
        return len(self.train_data)



def make_dataset_clover_unlab(root):
    np.random.seed(42)
    img_paths = []
    with open(os.path.join(root,"train.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            if row[-1] == "missing data":
                continue
            img_paths.append(os.path.join(root, 'images_downscaled', row[0].replace('npy','jpg').replace('pred','')))
            
    return img_paths

def make_dataset_clover(root):
    np.random.seed(42)
    img_paths = []
    files = [os.path.join(root, "train_red.csv")]
    for i, gt_file in enumerate(files):
        with open(gt_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i < 1:
                    continue
                else:
                    if row[-1] == "missing data" or float(row[1]) == 0:
                        continue
                    img_paths.append(os.path.join(root, 'images_downscaled', row[0].replace('JPG','jpg')))

    return img_paths
