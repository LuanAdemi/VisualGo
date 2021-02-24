# a custom class for image transformations

import numpy as np
import cv2
from scipy.spatial import distance as dist

from skimage.transform import resize

import torch

import imutils

import torchvision.transforms as T

# PerspectiveTransformer: Warps an image using a polygon mask
class PerspectiveTransformer:
    def __init__(self):
        self.output = np.float32([[0,0], [800-1,0], [800-1,800-1], [0,800-1]])
        
    def order_points(self, pts):

        xSorted = pts[np.argsort(pts[:, 0]), :]

        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
    
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        return np.array([tl, tr, br, bl], dtype="float32")
    
    def warp(self, img, points):
        matrix = cv2.getPerspectiveTransform(self.order_points(points), self.output)
        imgOutput = cv2.warpPerspective(img, matrix, (img.shape[0],img.shape[1]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        return imgOutput
    
    
# ThreasholdTransformer: get the black and white parts of an image
class ThreasholdTransformer:
    def filterBlack(self, img):
        ret_b, threash_b = cv2.threshold(img,40,255,cv2.THRESH_BINARY_INV)
        threash_b = cv2.cvtColor(threash_b, cv2.COLOR_BGR2GRAY)
        return threash_b
    
    def filterWhite(self, img):
        ret_w, threash_w = cv2.threshold(imgOutput,140,255,cv2.THRESH_BINARY)
        threash_w = cv2.cvtColor(threash_w, cv2.COLOR_BGR2GRAY)
        return threash_w
    
    
# MaskTransformer: perspectively warps an image using a mask prediction of the passed model
class MaskTransformer:
    def __init__(self, model, image_size):
        self.model = model
        
        self.pt = PerspectiveTransformer()
        self.tt = ThreasholdTransformer()
        
        self.imageSize = image_size
        
        self.imageTransformer = T.Compose([
            T.ToPILImage(),
            T.Resize(128),
            T.ToTensor(),
        ])
        
    def transform(self, img):
        assert img.shape[-1] == 3, "Image is not RGB. Abort"
        
        with torch.no_grad():
            img_t = self.imageTransformer(img).view(1,3,self.imageSize,self.imageSize)
        
            predMask = (self.model(img_t).cpu()[0][0].numpy() < 1)
            
            predMask = resize(predMask, (800,800)).astype(np.uint8)
            
            cnts = cv2.findContours(predMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
            
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # if our approximated contour has four points, then we
                # can assume that we have found our screen
                if len(approx) == 4:
                    screenCnt = approx
                    break

            res = self.pt.warp(img, screenCnt.reshape(4,2))
            
        return res, predMask, screenCnt