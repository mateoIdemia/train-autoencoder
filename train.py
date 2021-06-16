import wandb
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from cae_32x32x32 import CAE
import holocron
from trainer import AutoencoderTrainer
from ssim import SSIM
import math
from lasink_simulation_dataset import LasinkSimulation
from typing import Optional, Dict
from dncnn import DnCNN

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)


def build_dataset(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    img_size = config['image_size']
    train_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(size=img_size, scale=(0.24,0.25)),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                        transforms.RandomHorizontalFlip()
                    ])

    val_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(size=img_size, scale=(0.24,0.25))

                    ])

    dsTrain = LasinkSimulation('data/nvidia_v2/train/', train_transforms)
    dsVal = LasinkSimulation('data/nvidia_v2/test/', val_transforms)

    train_loader = DataLoader(dsTrain, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(dsVal, batch_size=config['batch_size'], shuffle=True)

    return train_loader, val_loader


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader, val_loader = build_dataset(config)
        model = DnCNN(channels=3)
        model.apply(weights_init_kaiming)
        lr = config['lr']
        wd = config['wd']

        # Loss function
        criterion = SSIM()


        # Create the contiguous parameters.
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = holocron.optim.RAdam(model_params, lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=wd)


        #Trainer
        trainer = AutoencoderTrainer(model, train_loader, val_loader, criterion, optimizer, 0, output_file=config['checkpoint'], configwb=True)

        trainer.fit_n_epochs(config['epochs'], config['lr'], config['freeze'])
