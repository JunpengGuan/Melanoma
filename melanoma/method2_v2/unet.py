from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Classic U-Net (CNN encoder–decoder + skips). Binary segmentation: 1 logit channel."""

    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.down1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)
        self.mid = DoubleConv(c4, c4 * 2)
        self.up4 = nn.ConvTranspose2d(c4 * 2, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x = self.pool1(x1)
        x2 = self.down2(x)
        x = self.pool2(x2)
        x3 = self.down3(x)
        x = self.pool3(x3)
        x4 = self.down4(x)
        x = self.pool4(x4)
        x = self.mid(x)
        x = self.up4(x)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        return self.out_conv(x)


def build_unet() -> UNet:
    return UNet(in_ch=3, base=32)
