import torch
import torch.nn as nn
import timm
import torch.nn.functional as F_nn

class DepthModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b4', pretrained=True):
        super().__init__()
        
        # --------------------------
        # Encoder: pretrained CNN
        # --------------------------
        # Use features_only=True to get intermediate feature maps for decoder
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        
        # Channels of encoder feature maps at each stage
        encoder_channels = self.encoder.feature_info.channels()  # e.g., [48, 56, 160, 448]
        last_ch = encoder_channels[-1]

        # --------------------------
        # Simple decoder: upsample to original resolution
        # --------------------------
        self.decoder = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output: single-channel depth map
        )

    def forward(self, x):
        # Encoder forward: returns list of feature maps at different stages
        input_size = x.shape[2:]  # height and width of **original input image**
        
        features = self.encoder(x)
        x_enc = features[-1]
        
        depth = self.decoder(x_enc)
        depth = F_nn.interpolate(depth, size=input_size, mode='bilinear', align_corners=False)
        
        return depth