import os
import random

import torch
import torch.utils.data as data
import SimpleITK as sitk
import numpy as np
import math
from skimage.transform import resize
from scipy.ndimage import zoom
from DataSet.ImageUtils import *
from scipy.ndimage import zoom

from collections import Counter

class ImageFolder(data.Dataset):
    def __init__(self, args, split, crop_size=(64, 192, 192), scale = True):
        self.root = args.root
        self.scale = scale
        self.classes = args.num_classes
        self.files = read_data_paths_resize(self.root, split)
        self.crop_d, self.crop_h, self.crop_w = crop_size

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = read_datafromITK(datafiles["image_path"])
        
        #image = image.astype(np.float32)
        label = read_datafromITK(datafiles["label_path"])
        label[label>=self.classes] = self.classes-1 # 去肿瘤做单器官分割
        if label.shape[0] == label.shape[1] and label.shape[1] != label.shape[2]:
            image = np.transpose(image, (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))

        # print(datafiles["name"],Counter(label.flatten()))
        # tumor_label 避免tumor label丢失
        if image.shape[0] != self.crop_d or image.shape[1] != self.crop_h or image.shape[2]!=self.crop_w:
            image = zoom(image, (self.crop_d/image.shape[0], self.crop_h/image.shape[1], self.crop_w/image.shape[2]), order=1)
            if self.classes>2:
                tumor_label = label.copy()
                tumor_label[tumor_label<2] = 0
                tumor_label = zoom(tumor_label, (self.crop_d / tumor_label.shape[0], self.crop_h / tumor_label.shape[1], self.crop_w / tumor_label.shape[2]), order=3)
            label = zoom(label, (self.crop_d/label.shape[0], self.crop_h/label.shape[1], self.crop_w/label.shape[2]), order=0)
            if self.classes>2:
                label[tumor_label>0] = 2
        # print(datafiles["name"], Counter(label.flatten()))
        # 标准化
        min_n = np.min(image)
        max_n = np.max(image)
        image = (image - min_n) / (max_n-min_n)        
        image = torch.FloatTensor(image).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)

        return image, label.squeeze(0), datafiles["name"]

    def __len__(self):
        return len(self.files)

class ImagePredFolder(data.Dataset):
    def __init__(self, args, split, crop_size=(64, 192, 192), scale = True):
        self.root = args.root
        self.scale = scale
        self.classes = args.num_classes
        self.files = read_pred_data(self.root, split)
        self.crop_d, self.crop_h, self.crop_w = crop_size
    def __getitem__(self, index):
        datafiles = self.files[index]
        image = read_datafromITK(datafiles["image_path"])
        if image.shape[0] != self.crop_d or image.shape[1] != self.crop_h or image.shape[2]!=self.crop_w:
            image = zoom(image, (self.crop_d/image.shape[0], self.crop_h/image.shape[1], self.crop_w/image.shape[2]), order=1)
        min_n = np.min(image)
        max_n = np.max(image)
        image = (image - min_n) / (max_n-min_n)        
        image = torch.FloatTensor(image).unsqueeze(0)
        return image, datafiles["name"]

    def __len__(self):
        return len(self.files)