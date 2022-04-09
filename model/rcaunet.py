import torch
from torch import nn, cat
import numpy as np
from .st_lstm import Prior_STLSTM


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(in_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(out_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(out_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )


class CABlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, reduction=16):
        super(CABlock, self).__init__()

        self.conv_block = conv_block(in_channels, out_channels)

        self.attetion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        y = self.attetion(x)
        out = x * y
        return out


class CAUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CAUNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, out_channels, 3, padding=0))

    def enc(self, x):
        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)
        return x, [conv1, conv2, conv3, conv4]
    
    def dec(self, low_features, enc_list):
        conv1, conv2, conv3, conv4 = enc_list
        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        x = x - conv1

        out = self.last_conv(x)
        return out

    def forward(self, x):
        h, w = x.shape[2:]
        target = 32
        h, w = h//target*target, w//target*target
        x = x[..., :h, :w]

        low_features, enc_list = self.enc(x)
        low_features = self.inter_conv(low_features)
        out = self.dec(low_features, enc_list)
        illumination = out.mean((2, 3))
        return illumination

class ReCurrentCAUnet(CAUNet):
    def __init__(self, in_channels=3, out_channels=3):
        super(CAUNet, self).__init__(in_channels=in_channels, out_channels=out_channels)
        self.st_lstm = Prior_STLSTM(256, 256)
    
    def forward(self, x, lstm_memory):
        h, c, M = lstm_memory
        x, enc_list = self.enc(x)
        st_fused_info, h, c, M = self.st_lstm(x, h, c, M)
        low_features = self.inter_conv(st_fused_info)
        out = self.dec(low_features, enc_list)
        return out
