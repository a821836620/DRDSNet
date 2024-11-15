import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class D3PMLoss(nn.Module):
    def __init__(self):
        super(D3PMLoss, self).__init__()

    def forward(self, generated, target, segmentation_output, segmentation_target):
        # DDPM生成损失（MSE）
        mse_loss = F.mse_loss(generated, target)
        
        # 分割损失（Dice系数损失）
        dice_loss = self.dice_loss(segmentation_output, segmentation_target)
        
        return mse_loss + dice_loss

    def dice_loss(self, predicted, target):
        smooth = 1e-5
        intersection = (predicted * target).sum()
        return 1 - (2. * intersection + smooth) / (predicted.sum() + target.sum() + smooth)