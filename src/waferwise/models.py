"""Model definitions."""

import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 conv -> batch norm -> ReLU -> 2x2 max pool (halves the spatial size)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WaferCNN(nn.Module):
    """Small CNN for 1 x 64 x 64 wafer maps.

    Four conv blocks (64 -> 32 -> 16 -> 8 -> 4 pixels), global average pooling
    and a linear classifier. About 100k parameters, so it trains on a laptop CPU
    and stays fast after INT8 export.
    """

    def __init__(self, num_classes: int, channels=(16, 32, 64, 128), dropout: float = 0.3, in_channels: int = 1):
        super().__init__()
        blocks, prev = [], in_channels
        for ch in channels:
            blocks.append(conv_block(prev, ch))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))


def build_model(model_cfg: dict, num_classes: int) -> WaferCNN:
    return WaferCNN(
        num_classes=num_classes,
        channels=tuple(model_cfg.get("channels", (16, 32, 64, 128))),
        dropout=model_cfg.get("dropout", 0.3),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
