import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import torchvision.models as models
from torch.nn import ModuleList

def predict_disp(in_channels):
    return nn.Sequential(
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels, out_channels=1, kernel_size=3),
        nn.Sigmoid()
    )

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.LeakyReLU(inplace=True)
    )

class DispNetS_v2(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispNetS_v2, self).__init__()

        self.alpha = alpha
        self.beta = beta

        encoder_channels = [64, 64, 128, 256, 512]
        resnet = models.resnet18(pretrained=True)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        decoder_channels = [256, 128, 64, 32, 16]
        self.decoder5 = ModuleList([
            conv(encoder_channels[-1], decoder_channels[0]),
            conv(decoder_channels[0] + encoder_channels[-2], decoder_channels[0])
        ])
        self.decoder4 = ModuleList([
            conv(decoder_channels[0], decoder_channels[1]),
            conv(decoder_channels[1] + encoder_channels[-3], decoder_channels[1])
        ])
        self.decoder3 = ModuleList([
            conv(decoder_channels[1], decoder_channels[2]),
            conv(decoder_channels[2] + encoder_channels[-4], decoder_channels[2])
        ])
        self.decoder2 = ModuleList([
            conv(decoder_channels[2], decoder_channels[3]),
            conv(decoder_channels[3] + encoder_channels[-5], decoder_channels[3])
        ])
        self.decoder1 = ModuleList([
            conv(decoder_channels[3], decoder_channels[4]),
            conv(decoder_channels[4], decoder_channels[4])
        ])

        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")

        self.predict_disp4 = predict_disp(decoder_channels[-4])
        self.predict_disp3 = predict_disp(decoder_channels[-3])
        self.predict_disp2 = predict_disp(decoder_channels[-2])
        self.predict_disp1 = predict_disp(decoder_channels[-1])

        
    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             zeros_(m.bias)
        pass

    def forward(self, x):
        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(out_encoder1)
        out_encoder3 = self.encoder3(out_encoder2)
        out_encoder4 = self.encoder4(out_encoder3)
        out_encoder5 = self.encoder5(out_encoder4)

        out_decoder5 = self.decoder5[1](torch.cat([self.upsample(self.decoder5[0](out_encoder5)), out_encoder4], dim=1))
        out_decoder4 = self.decoder4[1](torch.cat([self.upsample(self.decoder4[0](out_decoder5)), out_encoder3], dim=1))
        disp4 = self.alpha * self.predict_disp4(out_decoder4) + self.beta
        out_decoder3 = self.decoder3[1](torch.cat([self.upsample(self.decoder3[0](out_decoder4)), out_encoder2], dim=1))
        disp3 = self.alpha * self.predict_disp3(out_decoder3) + self.beta
        out_decoder2 = self.decoder2[1](torch.cat([self.upsample(self.decoder2[0](out_decoder3)), out_encoder1], dim=1))
        disp2 = self.alpha * self.predict_disp2(out_decoder2) + self.beta
        out_decoder1 = self.decoder1[1](self.upsample(self.decoder1[0](out_decoder2)))
        disp1 = self.alpha * self.predict_disp1(out_decoder1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1
