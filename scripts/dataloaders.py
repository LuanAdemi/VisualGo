import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
import cv2

from tqdm import tqdm, tqdm_notebook
from skimage.draw import polygon2mask, polygon, polygon_perimeter

from scripts.transformers import MaskTransformer, PerspectiveTransformer, ThreasholdTransformer

import torch
import torchvision.transforms as T


# define a dataset class for the Dataloaders
class MaskDataset:
    def __init__(self, baseFile , folder, image_size):
        self.basePath = folder
        self.imageSize = image_size
        
        # load the dataset
        with open(baseFile) as json_file:
            self.jsonData = json.load(json_file)
        
        self.images = []
        
        self.masks = []
        
        # save every image and ground truth mask
        for p in tqdm(self.jsonData):
            data = p["data"]
            imagePath = data["image"]
        
            self.images.append(np.array(Image.open(self.basePath + os.path.basename(imagePath))))
            
            y = np.array(p["completions"][0]["result"][0]["value"]["points"])
            mask = polygon2mask((800,800), y*8).astype(bool).T
            
            self.masks.append(mask)
    
    def __len__(self):
        return len(self.jsonData)
    
    def __getitem__(self, index):
        
        # a simple image transformer wich resizes the images
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.imageSize),
            T.ToTensor(),
        ])
        
        img = self.images[index]
        X = transform(img)
        
        mask = resize(self.masks[index], (self.imageSize,self.imageSize), order = 0,preserve_range=True)
        
        return X, torch.LongTensor(mask)
    
    
class PositionDataset:
    def __init__(self, folder, image_size, maskmodel):
        self.basePath = folder
        self.imageSize = image_size
        
        # load out UNet model, trained for masking the board images
        self.maskmodel = maskmodel
        
        # define our transformers
        self.mt = MaskTransformer(self.maskmodel, self.imageSize)
        self.pt = PerspectiveTransformer()
        self.tt = ThreasholdTransformer()
        
        self.images = []
        self.target = []
        
        # save every image and ground truth mask
        for i in tqdm(range(len(os.listdir(folder))//2)):
            image = f"board_{i}.jpg"
            gt = f"board_{i}.csv"
        
            img = (np.array(Image.open(self.basePath + image)))
            
            try:
                img_t, mask, poly = self.mt.transform(img)
                self.images.append(img_t)
                y = np.genfromtxt(self.basePath + gt,delimiter=',')
            
                self.target.append(y)
            except:
                continue
            
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        # a simple image transformer wich resizes the images
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.imageSize),
            T.ToTensor(),
        ])
        
        img = self.images[index]
        X = transform(img)
        
        target = self.target[index]
        
        return X, torch.LongTensor(target)