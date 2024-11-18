import os
import random

import torch
import torch.utils.data as data
import SimpleITK as sitk
import numpy as np
import math
from skimage.transform import resize
from scipy.ndimage import zoom
from DataSets.ImageUtils import *
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
        images = []
        for img_path in os.listdir(datafiles["image_path"]):
            if img_path.endswith('.nii'):
                image = read_datafromITK(os.path.join(datafiles["image_path"], img_path))
                images.append(image)
        #image = image.astype(np.float32)
        label = read_datafromITK(datafiles["label_path"])
        if label.shape[0] == label.shape[1] and label.shape[1] != label.shape[2]:
            images = [np.transpose(x, (2, 0, 1)) for x in images]
            label = np.transpose(label, (2, 0, 1))

        # print(datafiles["name"],Counter(label.flatten()))
        # tumor_label 避免tumor label丢失
        if label.shape[0] != self.crop_d or label.shape[1] != self.crop_h or label.shape[2]!=self.crop_w:
            images = [zoom(x, (self.crop_d/x.shape[0], self.crop_h/x.shape[1], self.crop_w/x.shape[2]), order=1) for x in images]
            if self.classes>2:
                tumor_label = label.copy()
                tumor_label[tumor_label<2] = 0
                tumor_label = zoom(tumor_label, (self.crop_d / tumor_label.shape[0], self.crop_h / tumor_label.shape[1], self.crop_w / tumor_label.shape[2]), order=3)
            label = zoom(label, (self.crop_d/label.shape[0], self.crop_h/label.shape[1], self.crop_w/label.shape[2]), order=0)
            if self.classes>2:
                label[tumor_label>0] = 2
        # print(datafiles["name"], Counter(label.flatten()))
        # 标准化
        for i in range(len(images)):
            min_n = np.min(images[i])
            max_n = np.max(images[i])
            images[i] = (images[i] - min_n) / (max_n-min_n)
            images[i] = torch.FloatTensor(images[i]).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)

        return images, label.squeeze(0), datafiles["name"]

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
        image = read_datafromITK(os.path.join(datafiles["image_path"], 'V0.nii.gz'))
        if image.shape[0] != self.crop_d or image.shape[1] != self.crop_h or image.shape[2]!=self.crop_w:
            image = zoom(image, (self.crop_d/image.shape[0], self.crop_h/image.shape[1], self.crop_w/image.shape[2]), order=1)
        min_n = np.min(image)
        max_n = np.max(image)
        image = (image - min_n) / (max_n-min_n)        
        image = torch.FloatTensor(image).unsqueeze(0)
        return image, datafiles["name"]

    def __len__(self):
        return len(self.files)