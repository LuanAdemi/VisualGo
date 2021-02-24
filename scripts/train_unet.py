#!/usr/bin/env python
# coding: utf-8

# # Predicting the position of a go board in an image using UNet
# 
# This notebook uses the UNet architecture to create a heatmap from an image containing a go board.

# In[1]:


import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
from tqdm import tqdm

from scipy.spatial import distance as dist

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from skimage.transform import resize

import cv2

from tqdm import tqdm, tqdm_notebook

from skimage.draw import polygon2mask, polygon, polygon_perimeter

from adabelief_pytorch import AdaBelief

import torch
import torch.nn.functional as F
import torch.nn as nn
from scripts.unet import UNet
from script.dataloaders import MaskDataset

datafolder = "data/board_masks/upload/"


# our dataset
print('#'*100)
print("Loading dataset")
d = MaskDataset("data/result.json",datafolder, 128)


# In[7]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=2, wf=5, depth=4, padding=True, up_mode='upsample').to(device)
optim = AdaBelief(model.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decouple = True, rectify = False)


# In[8]:


batch_size = 16
train_dl = DataLoader(d, batch_size, shuffle=True, num_workers=8,)


# In[9]:

print('#'*100)
print("Training")
epochs = 150
batches = len(train_dl)

for epoch in range(epochs):
    total_loss = 0
    progress = tqdm(enumerate(train_dl), desc="Loss: ", total=batches)
    
    model.train()
    
    for i, (X, target) in progress:
        X = X.to(device)  # [N, 3, H, W]
        target = target.to(device)  # [N, H, W] with class indices (0, 1)
        
        outputs = model(X) # [N, 1, H, W]
        
        loss = F.cross_entropy(outputs, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        current_loss = loss.item()
        total_loss += current_loss
        
        progress.set_description(f"Epoch: {epoch} | Loss: {(total_loss/(i+1))}")
        
    torch.cuda.empty_cache() 
    val_losses = 0


# In[19]:


torch.save(model.state_dict(), 'state_dicts/checkpoint.pth')

