import torch
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
# target one-hot编码
from torch.autograd._functions import tensor
from collections import  Counter

from sklearn import manifold
import matplotlib.pyplot as plt

def get_tsne(data, n_components = 2, n_images = None):
    if n_images is not None:
        data = data[:n_images]
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

def plot_representations(train_label, data, labels, epoch, save_path, n_images = None,):
            
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
                
    # fig = plt.figure(figsize = (10, 10),dpi=600)
    fig = plt.figure(figsize = (15, 15), dpi=600)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c = labels, cmap = 'rainbow')
    a,b=scatter.legend_elements()
    b = []
    for label in train_label:
        b.append('$\\mathdefault{%s}$'%label)
    legend1 = ax.legend(a,b,loc="upper right", title="Classes")
    ax.add_artist(legend1)
    fig.savefig(save_path+'/%d.svg'%epoch)


def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]1
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot


class Save_Model():
    def __init__(self, save_path, num_classes):
        self.best_epoch = list(0 for i in range(num_classes))
        self.best_dice = list(0 for i in range(num_classes))
        self.num_classes = num_classes
        self.trigger = 0
        self.save_path = save_path

    def updata(self,epoch, state, val_log):
        self.trigger += 1
        torch.save(state, os.path.join(self.save_path, 'latent.pth'))
        for i in range(1, self.num_classes):
            if val_log['Val_dice_class%d'%i] > self.best_dice[i]:
                self.best_epoch[i] = epoch
                self.best_dice[i] = val_log['Val_dice_class%d'%i]
                self.trigger = 0
                print('Saving best model for class %d'%i)
                torch.save(state, os.path.join(self.save_path, 'best_class%d_model.pth'%i))
            print('Best class {} performance at Epoch: {} | {}'.format(i, self.best_epoch[i], self.best_dice[i]))


def merge(outputs, scaler, axis):
    assert len(outputs)%scaler==0
    patchs = []
    patch = None
    for idx, output in enumerate(outputs):
        if patch == None:
            patch = output
        elif idx%scaler ==0:
            patchs.append(patch)
            patch = output
        else:
            patch = torch.cat((patch, output), axis=axis)
    patchs.append(patch)
    return patchs

def patch_merge(outputs, scalers):
    cols = merge(outputs, scalers[-1],-1)
    rows = merge(cols, scalers[-2],-2)
    deps = merge(rows, scalers[-3],-3)
    assert len(deps) == 1
    return deps[0]

def save_results(args, outputs, case_names):
    print('start save')
    result_save_path = os.path.join(args.root, 'results')
    if not os.path.exists(os.path.join(result_save_path,'images')): os.makedirs(os.path.join(result_save_path,'images'))
    if not os.path.exists(os.path.join(result_save_path,'labels')): os.makedirs(os.path.join(result_save_path,'labels'))
    outputs = torch.argmax(outputs, dim=1).cpu()
    for i, case_name in enumerate(case_names):
        pred = np.asarray(outputs[i].numpy(), dtype='uint8')
        save_ = sitk.ReadImage(os.path.join(args.root, args.test_dir, 'images', case_name + '.nii.gz'), sitk.sitkInt16)
        img = sitk.GetArrayFromImage(save_)
        if img.shape != pred.shape:
            pred = zoom(pred, (img.shape[0]/pred.shape[0], img.shape[1]/pred.shape[1], img.shape[2]/pred.shape[2]), order=1)
            # img = zoom(img,
            #              (pred.shape[0] / img.shape[0], pred.shape[1] / img.shape[1], pred.shape[2] / img.shape[2]),
            #              order=1)
        pred = sitk.GetImageFromArray(pred)
        # pred.SetDirection(save_.GetDirection())
        # pred.SetOrigin(save_.GetOrigin())
        # pred.SetSpacing(save_.GetSpacing())

        img = sitk.GetImageFromArray(img)
        # img.SetDirection(save_.GetDirection())
        # img.SetOrigin(save_.GetOrigin())
        # img.SetSpacing(save_.GetSpacing())
        sitk.WriteImage(img, os.path.join(result_save_path,'images', case_name + '.nrrd'))
        sitk.WriteImage(pred, os.path.join(result_save_path,'labels', case_name + '.nrrd'))

