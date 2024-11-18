# 切片可视化
import os
import numpy as np
import cv2 as cv
from scipy.ndimage import zoom
import SimpleITK as sitk

def get_slice(path):
    slices = []
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    if image_array.shape[0] != 32:
        image_array = zoom(image_array, (32/image_array.shape[0], 256/image_array.shape[1], 256/image_array.shape[2]), order=1)
    for i in range(image_array.shape[0]):
        slices.append(image_array[i,:,:]*127)
    return slices

def save_result(merge_slices,save_path):
    for i in range(len(merge_slices)):
        cv.imwrite(os.path.join(save_path, '%d.png'%i),merge_slices[i])

def show_slice(path, target, model_names, case, label_path):
    save_path = os.path.join(target, case.strip('.nrrd'))
    if not os.path.exists(save_path): os.makedirs(save_path)
    # read label
    merge_slices = get_slice(os.path.join(label_path, case))
    for model_name in model_names:
        slices = get_slice(os.path.join(path,model_name,'labels',case))
        assert len(merge_slices)==len(slices)
        for i in range(len(slices)):
            merge_slices[i] = np.hstack((merge_slices[i], slices[i]))
    save_result(merge_slices, save_path)



