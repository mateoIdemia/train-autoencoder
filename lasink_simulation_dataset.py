import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import random
import numpy as np
from gen_fake_lasink import *
from torchvision.transforms import transforms

class LasinkSimulation(Dataset):


    def __init__(self, folder, transform=None):
        self.transforms=transform

        self.imgs = glob.glob(folder+'/*')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transforms_tensor = transforms.Compose([transforms.ToTensor(), normalize])  
                                    

    def __getitem__(self, index):
        
        img = Image.open(self.imgs[index])

        img = self.transforms(img)
        lasink = gen_fake_lasink(img, 1, 2)

        img = self.transforms_tensor(img)
        lasink = self.transforms_tensor(lasink)


        return img,  lasink

    def __len__(self):
        return len(self.imgs)
