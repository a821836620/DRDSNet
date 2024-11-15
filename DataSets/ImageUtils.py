import SimpleITK as sitk
import os
import numpy as np

def read_datafromITK(path):
    image = sitk.ReadImage(path, sitk.sitkInt16)
    image_array = sitk.GetArrayFromImage(image)
    return image_array

def read_data_paths_resize(root,split):
    files = []
    image_dir = os.path.join(root, split, 'images')
    label_dir = os.path.join(root, split, 'labels')
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name)
        files.append({
            "image_path": image_path,
            "label_path": label_path,
            "name": image_name.split('.')[0],
        })
    return files

def read_data_paths(root,split):
    files = []
    image_dir = os.path.join(root, split, 'images')
    label_dir = os.path.join(root, split, 'labels')
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name)
        label = read_datafromITK(label_path)
        bbox = []
        for i in range(1,np.max(label)+1):
            bx = np.where(label == i)
            bbox.append(bx)
        files.append({
            "image_path": image_path,
            "label_path": label_path,
            "name": image_name.split('.')[0],
            "bbox": bbox
        })
    return files

def read_pred_data(root,split):
    files = []
    image_dir = os.path.join(root, split, 'images')
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        files.append({
            "image_path": image_path,
            "name": image_name.split('.')[0]
        })
    return files