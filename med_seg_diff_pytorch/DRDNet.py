import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_channels, attention_dim=64):
        super(UNet3D, self).__init__()
        
        # 编码器（下采样部分）
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 1024),
        ])
        
        # 解码器（上采样部分）
        self.decoder = nn.ModuleList([
            self.deconv_block(1024, 512),
            self.deconv_block(512, 256),
            self.deconv_block(256, 128),
            self.deconv_block(128, 64),
            nn.Conv3d(64, out_channels, kernel_size=1),
        ])
        
        # 交叉注意力层（每层下采样与HTSF特征进行融合）
        self.attn_blocks = nn.ModuleList([
            CrossAttentionBlock(64, temporal_channels, attention_dim),
            CrossAttentionBlock(128, temporal_channels, attention_dim),
            CrossAttentionBlock(256, temporal_channels, attention_dim),
            CrossAttentionBlock(512, temporal_channels, attention_dim),
            CrossAttentionBlock(1024, temporal_channels, attention_dim),
        ])
    
    def conv_block(self, in_channels, out_channels):
        """定义卷积块，包含两个卷积层、BatchNorm和ReLU激活"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def deconv_block(self, in_channels, out_channels):
        """定义反卷积块，包含一个转置卷积层和激活函数"""
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, hemodynamic_features):
        # 编码器部分
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        enc5 = self.encoder[4](enc4)
        
        # 交叉注意力融合（每一层编码器特征与HTSF特征融合）
        enc1 = self.attn_blocks[0](enc1, hemodynamic_features)
        enc2 = self.attn_blocks[1](enc2, hemodynamic_features)
        enc3 = self.attn_blocks[2](enc3, hemodynamic_features)
        enc4 = self.attn_blocks[3](enc4, hemodynamic_features)
        enc5 = self.attn_blocks[4](enc5, hemodynamic_features)
        
        # 解码器部分
        dec4 = self.decoder[0](enc5)
        dec3 = self.decoder[1](dec4 + enc4)  # Skip connections
        dec2 = self.decoder[2](dec3 + enc3)
        dec1 = self.decoder[3](dec2 + enc2)
        out = self.decoder[4](dec1 + enc1)
        
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, spatial_channels, temporal_channels, attention_dim=64):
        super(CrossAttentionBlock, self).__init__()
        # 使用Multihead Attention来进行时空特征融合
        self.attn = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=4)
        
        # 用于空间和时间特征的投影层
        self.spatial_proj = nn.Linear(spatial_channels, attention_dim)
        self.temporal_proj = nn.Linear(temporal_channels, attention_dim)
        
    def forward(self, spatial_features, temporal_features):
        """
        Args:
            spatial_features (Tensor): 当前空间特征 (B, C, D, H, W)
            temporal_features (Tensor): HTSF的时间特征 (B, T, C, D, H, W)
        """
        # 对空间和时间特征进行投影以适配多头自注意力
        B, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, C, -1).permute(2, 0, 1)  # (D*H*W, B, C)
        temporal_features = temporal_features.view(B, -1, C, D, H, W).permute(1, 0, 2, 3, 4)  # (T, B, C, D, H, W)
        temporal_features = temporal_features.flatten(start_dim=3).permute(1, 0, 2)  # (B*T, C, D*H*W)
        
        # 对空间和时间特征进行投影
        spatial_proj = self.spatial_proj(spatial_features).permute(2, 0, 1)  # (D*H*W, B, attention_dim)
        temporal_proj = self.temporal_proj(temporal_features).permute(2, 0, 1)  # (B*T, C, attention_dim)
        
        # 进行多头注意力
        attn_output, _ = self.attn(spatial_proj, temporal_proj, temporal_proj)
        
        # 恢复到空间维度
        output = attn_output.permute(1, 2, 0).view(B, C, D, H, W)
        return output


class D3PM(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_channels, timesteps=1000):
        super(D3PM, self).__init__()
        
        # 3D U-Net 模型
        self.unet = UNet3D(in_channels, out_channels, temporal_channels)
        
        # 定义DDPM相关参数
        self.timesteps = timesteps
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def forward(self, x, hemodynamic_features):
        # 生成每个时间步的DCE-MRI序列
        noise = torch.randn_like(x)  # 初始噪声
        for t in range(self.timesteps):
            # 在每个时间步生成一个噪声图像
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            # 模拟前向扩散过程
            x_t = alpha_cumprod_t * x + (1 - alpha_cumprod_t) * noise
            
            # 使用3D U-Net生成图像
            generated_img = self.unet(x_t, hemodynamic_features)
            
            # 返回生成的图像
            return generated_img