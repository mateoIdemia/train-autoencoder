import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import random
import numpy as np

class LasinkSimulation(Dataset):


    def __init__(self, folder, transform=None):
        self.transforms=transform

        self.imgs = glob.glob(folder+'/original/*')
        self.matrix = glob.glob(folder+'/matrix/*')
            

    def __getitem__(self, index):
        
        img = Image.open(self.imgs[index])
        mat = Image.open(self.matrix[index])

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transforms(img)
        random.seed(seed)
        torch.manual_seed(seed)
        mat = self.transforms(mat)

        return img,  mat

    def __len__(self):
        return len(self.imgs)
