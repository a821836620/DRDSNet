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
from Models import *

import config
from Utils import weights_init, logger, metrics, common
from Loss3D import *

# dynamic blending weight
def dynamic_blending_weight(epoch, total_epochs, delta=0.8):
    return max(0, delta * (1 - epoch / total_epochs))

def train(model, loader, ddpm_optimizer, seg_optimizer, seg_criterion, ddpm_criterion, device, T, epoch):
    model.train()
    running_loss = 0.0
    for mris, label, case_name in tqdm(loader, desc="Training"):
        mris, label = [x.to(device) for x in mris], label.to(device)
        running_loss
        if epoch < 200:
            ddpm_optimizer.zero_grad()
            ddpm_loss = 0
            generated_mri = mris[0]
            for t, mri in enumerate(mris):
                beta = dynamic_blending_weight(epoch, 750)
                blended_mri = beta * label + (1 - beta) * generated_mri
                generated_mri, _ = model.ddpm_with_features(blended_mri, t)
                loss = ddpm_criterion(generated_mri, mri)
                ddpm_loss += loss
            ddpm_loss.backward()
            ddpm_optimizer.step()
            running_loss += ddpm_loss
        else:
            ddpm_optimizer.zero_grad()
            seg_optimizer.zero_grad()
            beta = dynamic_blending_weight(epoch, 750)
            seg_output, generated_mris = model(mris[0], T, beta, mris[1:])
            # first update ddpm
            ddpm_loss = 0
            for i, generated_mri in enumerate(generated_mris):
                loss = ddpm_criterion(generated_mri, mris[i+1])
                ddpm_loss += loss
            ddpm_loss.backward()
            ddpm_optimizer.step()
            # update segnet
            seg_loss = seg_criterion(seg_output, label)
            ce_loss = nn.BCEWithLogitsLoss()(seg_output, label)
            seg_loss += ce_loss
            seg_loss.backward()
            seg_optimizer.step()
            running_loss = running_loss + ddpm_loss.item() + seg_loss.item()

    return running_loss / len(loader)


def validate(model, loader, seg_criterion, ddpm_criterion, device, T):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mris, label, case_name in tqdm(loader, desc="Validation"):
            mris, label = [x.to(device) for x in mris], label.to(device)
            seg_output, generated_mris = model(mris[0], T)
            ddpm_loss = 0
            for i, generated_mri in enumerate(generated_mris):
                loss = ddpm_criterion(generated_mri, mris[i+1])
                ddpm_loss += loss
            seg_loss = seg_criterion(seg_output, label)
            ce_loss = nn.BCEWithLogitsLoss()(seg_output, label)
            seg_loss += ce_loss
            val_loss = val_loss+seg_loss.item() + ddpm_loss.item()
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
    ddpm_model = DDPM3D(in_channels=1, out_channels=1, temporal_channels=args.T).to(device)
    seg_model = UNet(c_in=1, c_out=1, time_dim=256, device="cpu")
    model = DRDSNet(ddpm=ddpm_model, unet_seg=seg_model, num_steps=4,).to(device)
    t_seq = torch.tensor([100, 200, 300, 400])  # DDPM time steps
    seg_criterion = D3PMLoss()  # 
    ddpm_optimizer = optim.Adam(model.ddpm.parameters(), lr=1e-4)
    seg_optimizer = optim.Adam(model.seg_net.parameters(), lr=1e-4)
    ddpm_criterion = nn.MSELoss()  # DDPM use MSE loss

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

    blending_total_epochs = int(0.75 * args.epochs)  # dynamic blending weight total epochs

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train(model, train_loader, ddpm_optimizer, seg_optimizer, seg_criterion, ddpm_criterion, device, args.T, epoch)
        val_loss = validate(model, val_loader, seg_criterion, ddpm_criterion, device, args.T)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")





