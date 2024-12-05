import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import torchvision.models as models


class ConvBlock(nn.Module):
    """A basic convolutional block: Conv2D + ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv


class Conv3x3(nn.Module):
    """3x3 convolution with padding."""
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv


class ResnetEncoder(nn.Module):
    """ResNet encoder for depth estimation."""
    def __init__(self, num_layers=18, pretrained=True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])  # ResNet18默认的通道数
        resnets = {18: models.resnet18, 34: models.resnet34}
        if num_layers not in resnets:
            raise ValueError("Only ResNet18 or ResNet34 are supported.")
        self.encoder = resnets[num_layers](pretrained)

    def forward(self, x):
        """提取多层特征"""
        features = []
        x = (x - 0.45) / 0.225  # 正则化
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))  # 第一层特征
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        return features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()

        for i in range(4, -1, -1):  # Decoder layers
            # upconv_0
            num_ch_in = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:  # Disparity prediction layers
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]

        for i in range(4, -1, -1):  # Iterate from deepest to shallowest layer
            x = self.convs[("upconv", i, 0)](x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # 保证 x 始终为张量
            if self.use_skips and i > 0:
                x = torch.cat((x, input_features[i - 1]), dim=1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return outputs


class DispNetS_v2(nn.Module):
    """DispNetS_v2: Combines ResNet encoder and DepthDecoder."""
    def __init__(self, num_layers=18, pretrained=True, scales=range(4), num_output_channels=1):
        super(DispNetS_v2, self).__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=scales, num_output_channels=num_output_channels)

    def forward(self, x):
        """Forward pass: encode features and decode depths."""
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs