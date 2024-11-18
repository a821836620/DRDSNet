import os
from collections import OrderedDict

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataSets.patch_dataset import PatchTrainFolder, PatchValFolder
from DataSets.dataset import ImageFolder, ImagePredFolder
from Models.DRDSNet import DRDSNet

import config
from Utils import weights_init, logger, metrics, common
from Loss3D import *

def test(model, loader, device, T):
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            x = batch['image'].to(device)
            pred = model(x, T)
            pred = torch.sigmoid(pred) > 0.5  # 二值化
            results.append((x.cpu(), pred.cpu()))
    return results


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    print('start')
    args = config.args
    in_size = list(int(x) for x in args.in_size.split(','))
    device = torch.device('cuda' if args.gpu_num>0 else 'cpu')

    checkpoint_path = args.test_path  


    # create model and load checkpoint
    model = DRDSNet(in_channels=1, out_channels=1, temporal_channels=args.T).to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    # data loader
    if args.data == 'patch':
        test_data = PatchTrainFolder(args.root,'test', in_size)
        
    elif args.data == 'image':
        test_data = ImageFolder(args.root,'test', in_size)
    else:
        raise ValueError('data error')
    
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # test
    results = test(model, test_loader, device, args.T)



