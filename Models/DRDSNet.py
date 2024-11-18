import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HTSF(nn.Module):
    def __init__(self, feature_dim, num_heads):
        """
        Initializes the HTSF module.
        Args:
            feature_dim: Dimension of the input features.
            num_heads: Number of attention heads for multi-head cross-attention.
        """
        super(HTSF, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Projection matrices for query, key, and value
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)

        # Attention normalization scaling factor
        self.scale = feature_dim ** 0.5

    def forward(self, spatial_feature, hemodynamic_features):
        """
        Performs the fusion of temporal hemodynamic features and spatial features.
        Args:
            spatial_feature (Tensor): Shape [B, C, H, W, D], spatial features from the segmentation network.
            hemodynamic_features (Tensor): Shape [T, B, C, H, W, D], temporal hemodynamic features.
        Returns:
            fused_feature (Tensor): Shape [B, C, H, W, D], fused feature map.
        """
        B, C, H, W, D = spatial_feature.shape
        T = hemodynamic_features.shape[0]

        # Flatten spatial features for attention calculation
        spatial_flat = spatial_feature.view(B, C, -1).permute(0, 2, 1)  # [B, HWD, C]
        
        # Process temporal features
        weighted_features = []
        for t in range(T):
            hemo_flat = hemodynamic_features[t].view(B, C, -1).permute(0, 2, 1)  # [B, HWD, C]

            # Calculate attention weights
            q = self.W_q(spatial_flat)  # [B, HWD, C]
            k = self.W_k(hemo_flat)    # [B, HWD, C]
            v = self.W_v(hemo_flat)    # [B, HWD, C]
            
            attn_weights = F.softmax(torch.bmm(q, k.transpose(-2, -1)) / self.scale, dim=-1)  # [B, HWD, HWD]
            weighted_hemo = torch.bmm(attn_weights, v)  # [B, HWD, C]

            # Reshape back to original spatial dimensions and add to list
            weighted_features.append(weighted_hemo.permute(0, 2, 1).view(B, C, H, W, D))

        # Aggregate weighted temporal features and fuse with spatial feature
        weighted_sum = sum(weighted_features) / T  # Temporal averaging
        fused_feature = weighted_sum + spatial_feature  # Add spatial feature

        return fused_feature


class CrossAttentionBlock(nn.Module):
    def __init__(self, spatial_channels, temporal_channels, attention_dim=64):
        super(CrossAttentionBlock, self).__init__()
        # use multi-head attention to fuse spatial and temporal features
        self.attn = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=4)
        
        # use linear layers to project spatial and temporal features to attention_dim
        self.spatial_proj = nn.Linear(spatial_channels, attention_dim)
        self.temporal_proj = nn.Linear(temporal_channels, attention_dim)
        
    def forward(self, spatial_features, temporal_features):
        """
        Args:
            spatial_features (Tensor): spatial features (B, C, D, H, W)
            temporal_features (Tensor): HTSF Temporal features (B, T, C, D, H, W)
        """
        # flatten spatial and temporal features
        B, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, C, -1).permute(2, 0, 1)  # (D*H*W, B, C)
        temporal_features = temporal_features.view(B, -1, C, D, H, W).permute(1, 0, 2, 3, 4)  # (T, B, C, D, H, W)
        temporal_features = temporal_features.flatten(start_dim=3).permute(1, 0, 2)  # (B*T, C, D*H*W)
        
        # spatial and temporal projection
        spatial_proj = self.spatial_proj(spatial_features).permute(2, 0, 1)  # (D*H*W, B, attention_dim)
        temporal_proj = self.temporal_proj(temporal_features).permute(2, 0, 1)  # (B*T, C, attention_dim)
        
        # multi-head cross-attention
        attn_output, _ = self.attn(spatial_proj, temporal_proj, temporal_proj)
        
        # reshape and return
        output = attn_output.permute(1, 2, 0).view(B, C, D, H, W)
        return output


class DRDSNet(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_channels, attention_dim=64):
        super(DRDSNet, self).__init__()

        # encoder
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 1024),
        ])

        # decoder
        self.decoder = nn.ModuleList([
            self.deconv_block(1024, 512),
            self.deconv_block(512, 256),
            self.deconv_block(256, 128),
            self.deconv_block(128, 64),
            nn.Conv3d(64, out_channels, kernel_size=1),
        ])

        # HTSF
        self.attn_blocks = nn.ModuleList([
            HTSF(feature_dim=64),
            HTSF(feature_dim=128),
            HTSF(feature_dim=256),
            HTSF(feature_dim=512),
            HTSF(feature_dim=1024),
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

    def forward(self, x, hemodynamic_features):
        # encoder
        enc_features = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            fused = self.attn_blocks[i](x, hemodynamic_features[i])
            enc_features.append(fused)
        
        # decoder
        dec = enc_features[-1]
        for i, dec_layer in enumerate(self.decoder[:-1]):
            dec = dec_layer(dec)
            if i < len(enc_features) - 1:
                dec = torch.cat([dec, enc_features[-(i + 2)]], dim=1)

        # output
        out = self.decoder[-1](dec)
        return out


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_channels, attention_dim=64):
        super(SegNet, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            self.deconv_block(512, 256),
            self.deconv_block(256, 128),
            self.deconv_block(128, 64),
            nn.Conv3d(64, out_channels, kernel_size=1),
        ])

        # HTSF modules
        self.attn_blocks = nn.ModuleList([
            HTSF(feature_dim=64),
            HTSF(feature_dim=128),
            HTSF(feature_dim=256),
            HTSF(feature_dim=512),
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

    def forward(self, x, hemodynamic_features):
        # Encoder
        enc_features = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            fused = self.attn_blocks[i](x, hemodynamic_features[i])
            enc_features.append(fused)

        # Decoder
        dec = enc_features[-1]
        for i, dec_layer in enumerate(self.decoder[:-1]):
            dec = dec_layer(dec)
            if i < len(enc_features) - 1:
                dec = torch.cat([dec, enc_features[-(i + 2)]], dim=1)

        # Output
        out = self.decoder[-1](dec)
        return out


import torch
import torch.nn as nn

class HTSF(nn.Module):
    def __init__(self, feature_dim):
        super(HTSF, self).__init__()
        self.Wq = nn.Linear(feature_dim, feature_dim)
        self.Wk = nn.Linear(feature_dim, feature_dim)
        self.Wv = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5

    def forward(self, spatial_feat, hemo_feats):
        fused_feats = []
        for t in range(len(hemo_feats)):
            hemo_feat = hemo_feats[t]
            q = self.Wq(spatial_feat)
            k = self.Wk(hemo_feat)
            v = self.Wv(hemo_feat)

            attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)
            fused = torch.matmul(attn_weights, v)
            fused_feats.append(fused + spatial_feat)

        return torch.stack(fused_feats, dim=0).mean(dim=0)

class DRDSNet(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_channels, attention_dim=64):
        super(DRDSNet, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
        ])

        # Decoder 
        self.decoder = nn.ModuleList([
            self.deconv_block(512, 256),
            self.deconv_block(256, 128),
            self.deconv_block(128, 64),
            nn.Conv3d(64, out_channels, kernel_size=1),
        ])

        # HTSF 
        self.attn_blocks = nn.ModuleList([
            HTSF(feature_dim=64),
            HTSF(feature_dim=128),
            HTSF(feature_dim=256),
            HTSF(feature_dim=512),
        ])

        # DDPM 
        self.noise_step = 1000
        self.noise_unet = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
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

    def generate_sequence(self, x, T):
        generated_sequence = []
        for t in range(T):
            noise = torch.randn_like(x)
            for i in reversed(range(self.noise_step)):
                x = x + noise  # Add noise
                for layer in self.noise_unet:
                    x = layer(x)
            generated_sequence.append(x)
        return generated_sequence

    def forward(self, x, T):
        # ddpm
        hemodynamic_features = [[] for _ in range(4)]  # save temporal features
        generated_sequence = self.generate_sequence(x, T)

        #  extract temporal features
        for t, mri_t in enumerate(generated_sequence):
            features = []
            x_temp = mri_t
            for i, enc in enumerate(self.encoder):
                x_temp = enc(x_temp)
                features.append(x_temp)

            for i in range(4):
                hemodynamic_features[i].append(features[i])

        #  stack temporal features
        for i in range(4):
            hemodynamic_features[i] = torch.stack(hemodynamic_features[i], dim=0)  # [T, C, D, H, W]

        #  HTSF
        enc_features = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            fused = self.attn_blocks[i](x, hemodynamic_features[i])
            enc_features.append(fused)

        #  decoder
        dec = enc_features[-1]
        for i, dec_layer in enumerate(self.decoder[:-1]):
            dec = dec_layer(dec)
            if i < len(enc_features) - 1:
                dec = torch.cat([dec, enc_features[-(i + 2)]], dim=1)

        #  output
        out = self.decoder[-1](dec)
        return out