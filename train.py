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


def train(model, loader, optimizer, criterion, device, T):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        x, y = batch['image'].to(device), batch['mask'].to(device)
        
        optimizer.zero_grad()
        pred = model(x, T)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)


def validate(model, loader, criterion, device, T):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            x, y = batch['image'].to(device), batch['mask'].to(device)
            pred = model(x, T)
            loss = criterion(pred, y)
            val_loss += loss.item()
    
    return val_loss / len(loader)


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

    seed_everything(args.seed)
    save_path = os.path.join('%s/%s/seed%d_%s_%s_lr%f_b%d_epo%d_opt%s_RLR%s_shape%s'%(args.save_path,args.work_name,args.seed, args.model_name, args.train_style, args.lr, args.batch_size, args.epochs, args.opt, args.ReduceLR, args.in_size))

    # log
    log = logger.Train_logger(save_dir=save_path, save_name='train_log')
    logger.save_args(save_path, args)  # 

    # create model
    model = DRDSNet(in_channels=1, out_channels=1, temporal_channels=args.T).to(device)
    criterion = D3PMLoss()  # 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # data loader
    if args.data == 'patch':
        train_data = PatchTrainFolder(args.root,'train', in_size)
        val_data = PatchValFolder(args.root,'train', in_size)
    elif args.data == 'image':
        train_data = ImageFolder(args.root,'train', in_size)
        val_data = ImagePredFolder(args.root,'train', in_size)
    else:
        raise ValueError('data error')
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device, args.T)
        val_loss = validate(model, val_loader, criterion, device, args.T)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")





