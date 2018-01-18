'''
Class for loading the Facades dataset
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os

class facadesDataset(Dataset):
    '''
        Facades Dataset
    '''

    def __init__(self, root_dir, transform=None):
        '''
        Args:
            root_dir (string): Directory with all the images
            transform(callable, optional): Optional transform to be applied on a sample
        '''

        self.root_dir = root_dir
        self.image_list = [f for f in os.listdir(self.root_dir) if (f.endswith('.png') or f.endswith('.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        width = image.size[0]
        height = image.size[1]
        img = image.crop((0, 0, width/2, height/2))
        gt = image.crop((width/2, height/2, width, height))

        if self.transform:
            img = self.transform(img)
            gt = self.transform(gt)

        return img, gt
