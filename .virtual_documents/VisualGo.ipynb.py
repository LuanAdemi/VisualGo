import os
from PIL import Image
import numpy as np
from numpy import genfromtxt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class Dataset:
    def __init__(self, folder, image_size):
        self.imgs = []
        self.gts = []
        self.image_size = image_size
        
        for i in range(len(os.listdir(folder))//2):
            self.imgs.append(np.array(Image.open(r""+folder+f"board_{i}.jpg")))
            self.gts.append(genfromtxt(folder+f"board_{i}.csv", delimiter=','))
            
        self.imgs = np.array(self.imgs)
        self.gts = np.array(self.gts)
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size),
            T.ToTensor()])
        
        image = self.imgs[index]
        X = transform(image)
        Y = self.gts[index].T
        return X, torch.FloatTensor(Y)


batch_size = 64
transformed_dataset = Dataset("goBoards/", 500)
train_dl = DataLoader(transformed_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([]) 
    ax.set_yticks([])
    ax.imshow(make_grid((images[0].detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for datapoint in dl:
        show_images(datapoint, nmax)
        break


show_batch(train_dl)



