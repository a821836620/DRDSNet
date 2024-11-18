import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Definition of the sinusoidal positional embedding layer
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# Definition of the UNet3D_DDPM model
class UNet3D_DDPM(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(UNet3D_DDPM, self).__init__()

        # Temporal MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
        ])

        # skip connection
        self.middle = self.conv_block(512, 512)

        # decoder
        self.decoder = nn.ModuleList([
            self.deconv_block(512, 256),
            self.deconv_block(256, 128),
            self.deconv_block(128, 64),
            nn.Conv3d(64, out_channels, kernel_size=1),
        ])

        # Temporal linear
        self.time_linear = nn.ModuleList([
            nn.Linear(time_emb_dim, 64),
            nn.Linear(time_emb_dim, 128),
            nn.Linear(time_emb_dim, 256),
            nn.Linear(time_emb_dim, 512),
        ])

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, timesteps):
        # time embedding
        time_emb = self.time_mlp(timesteps)

        # encoder
        enc_features = []
        for i, enc_layer in enumerate(self.encoder):
            x = enc_layer(x)
            enc_features.append(x + self.time_linear[i](time_emb).view(-1, enc_layer[0].out_channels, 1, 1, 1))

        # middle
        x = self.middle(x)

        # decoder
        for i, dec_layer in enumerate(self.decoder[:-1]):
            x = dec_layer(x)
            x = torch.cat([x, enc_features[-(i + 1)]], dim=1)  # 跳跃连接

        # output
        out = self.decoder[-1](x)
        return out

# Definition of the DDPM model
class DDPM(nn.Module):
    def __init__(self, model, beta_start=1e-4, beta_end=2e-2, num_timesteps=1000):
        super(DDPM, self).__init__()
        self.model = model
        self.num_timesteps = num_timesteps

        # Define the betas and alphas
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Define the square roots of the alphas and one minus alphas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def forward_diffusion(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1) * x_0 + \
              self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1) * noise
        return x_t, noise

    def reverse_diffusion(self, x_t, t):
        pred_noise = self.model(x_t, t)
        beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return (x_t - beta_t * pred_noise) / torch.sqrt(alpha_t_cumprod)

    def generate(self, shape, device):
        x_t = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x_t = self.reverse_diffusion(x_t, t)
        return x_t