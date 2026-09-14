"""
model.py — CNN-Swin Transformer hybrid model for SAR thickness prediction.

Architecture (reverse-engineered from cnn_swin_thickness_best.pth):
  - CNN branch:  ResNet-18 backbone (4 layers, BasicBlock, no final FC)
                 Output: 512-dim feature vector (after global avg pool)
  - Swin branch: Swin-Tiny (embed_dim=96, window_size=7, 4 stages → 768-dim)
                 Output: 768-dim feature vector (after norm + global avg pool)
  - Fusion:      Concat [512 + 768 = 1280] → Linear(1280, 512) → BN → ReLU
  - Head:        Linear(512, 3) — predicts 3 thickness values
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import timm


class CNNBranch(nn.Module):
    """ResNet-18 backbone stripped of the final FC layer."""

    def __init__(self):
        super().__init__()
        resnet = tvm.resnet18(weights=None)
        # Keep everything except avgpool and fc
        self.conv1    = resnet.conv1
        self.bn1      = resnet.bn1
        self.relu     = resnet.relu
        self.maxpool  = resnet.maxpool
        self.layer1   = resnet.layer1
        self.layer2   = resnet.layer2
        self.layer3   = resnet.layer3
        self.layer4   = resnet.layer4
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (B, 512)


class SwinBranch(nn.Module):
    """Swin-Tiny backbone using timm, output 768-dim features."""

    def __init__(self):
        super().__init__()
        # swin_tiny_patch4_window7_224 has embed_dim=96, 4 stages → final 768
        base = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0,    # remove classification head
        )
        # Expose named sub-modules that match checkpoint keys
        self.patch_embed = base.patch_embed
        self.layers      = base.layers
        self.norm        = base.norm

    def forward(self, x):
        x = self.patch_embed(x)          # (B, H, W, C)
        for layer in self.layers:
            x = layer(x)                 # (B, H', W', C')
        x = self.norm(x)                 # (B, 7, 7, 768) for swin-tiny
        # Global average pooling over spatial dims (H, W)
        x = x.mean(dim=[1, 2])           # (B, 768)
        return x


class CNNSwinThicknessModel(nn.Module):
    """Full hybrid model matching cnn_swin_thickness_best.pth."""

    def __init__(self, num_outputs: int = 3):
        super().__init__()
        self.cnn  = CNNBranch()           # → 512-dim
        self.swin = SwinBranch()          # → 768-dim

        # fusion_trunk matches checkpoint keys exactly
        self.fusion_trunk = nn.Sequential(
            nn.Linear(1280, 512),         # [0] weight/bias
            nn.BatchNorm1d(512),          # [1] weight/bias/running_*
            nn.ReLU(inplace=True),
        )

        self.thickness_head = nn.Linear(512, num_outputs)

    def forward(self, x):
        cnn_feat  = self.cnn(x)           # (B, 512)
        swin_feat = self.swin(x)          # (B, 768)
        fused     = torch.cat([cnn_feat, swin_feat], dim=1)   # (B, 1280)
        fused     = self.fusion_trunk(fused)                  # (B, 512)
        out       = self.thickness_head(fused)                # (B, 3)
        return out


def load_model(ckpt_path: str, device: str = "cpu") -> CNNSwinThicknessModel:
    """Load the trained thickness model from a checkpoint file."""
    model = CNNSwinThicknessModel(num_outputs=3)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Handle wrapped checkpoints
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in state_dict:
                state_dict = state_dict[key]
                break
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


# =========================================================================
# Stage 1 — Oil / No-Oil Detector
# =========================================================================
# Architecture (reverse-engineered from cnn_swin_v2_best.pth):
#   - CNN branch:  ResNet-18 backbone → 512-dim (same as thickness model)
#   - Swin branch: Swin-Tiny → 768-dim (same as thickness model)
#   - Fusion:      nn.Sequential:
#       [0] Linear(1280, 512)
#       [1] BatchNorm1d(512)
#       [2] ReLU
#       [3] Dropout(0.3)
#       [4] Linear(512, 128)
#       [5] ReLU
#       [6] Dropout(0.3)
#       [7] Linear(128, 2)   → 0 = No Oil, 1 = Oil
# =========================================================================

class CNNSwinOilDetector(nn.Module):
    """CNN-Swin hybrid for binary oil detection (Oil vs No-Oil).

    Uses the same ResNet-18 + Swin-Tiny backbone as the thickness model,
    but with a deeper fusion head that outputs 2 classes.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.cnn  = CNNBranch()           # → 512-dim
        self.swin = SwinBranch()          # → 768-dim

        # Deeper fusion head matching checkpoint key structure:
        #   fusion.0  = Linear(1280, 512)
        #   fusion.1  = BatchNorm1d(512)
        #   fusion.2  = ReLU
        #   fusion.3  = Dropout
        #   fusion.4  = Linear(512, 128)
        #   fusion.5  = ReLU
        #   fusion.6  = Dropout
        #   fusion.7  = Linear(128, 2)
        self.fusion = nn.Sequential(
            nn.Linear(1280, 512),         # [0]
            nn.BatchNorm1d(512),          # [1]
            nn.ReLU(inplace=True),        # [2]
            nn.Dropout(dropout),          # [3]
            nn.Linear(512, 128),          # [4]
            nn.ReLU(inplace=True),        # [5]
            nn.Dropout(dropout),          # [6]
            nn.Linear(128, 2),            # [7]
        )

    def forward(self, x):
        cnn_feat  = self.cnn(x)           # (B, 512)
        swin_feat = self.swin(x)          # (B, 768)
        fused     = torch.cat([cnn_feat, swin_feat], dim=1)  # (B, 1280)
        out       = self.fusion(fused)    # (B, 2)
        return out


def load_oil_detector(ckpt_path: str, device: str = "cpu") -> CNNSwinOilDetector:
    """Load the trained Oil/No-Oil detector from a checkpoint file."""
    model = CNNSwinOilDetector()
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Handle wrapped checkpoints
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in state_dict:
                state_dict = state_dict[key]
                break
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model

