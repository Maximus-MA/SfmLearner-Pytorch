import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import torchvision.models as models
from torch.nn import ModuleList


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.LeakyReLU(inplace=True)
    )


class PoseExpNet_v2(nn.Module):
    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(PoseExpNet_v2, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        # self.encoder0 = nn.Sequential(
        #     nn.Conv2d(3*(1+self.nb_ref_imgs), 3, kernel_size=1),
        #     nn.ReLU(inplace=True)
        # )

        encoder_channels = [64, 64, 128, 256, 512]
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        old_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(3*(1+self.nb_ref_imgs), 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data[:, :3] = old_conv1.weight.data
        self.encoder1 = nn.Sequential(
            new_conv1,
            resnet.bn1, 
            resnet.relu
        )
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        self.pose_pred = nn.Conv2d(encoder_channels[-1], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        # self.pretrained_modules = ModuleList([self.encoder2, self.encoder3, self.encoder4, self.encoder5])

        if self.output_exp:
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

            self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear")

            self.predict_mask4 = nn.Conv2d(decoder_channels[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(decoder_channels[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(decoder_channels[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(decoder_channels[4], self.nb_ref_imgs, kernel_size=3, padding=1)


    def init_weights(self):
        pass

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)

        # input = self.encoder0(input)
        out_encoder1 = self.encoder1(input)
        out_encoder2 = self.encoder2(out_encoder1)
        out_encoder3 = self.encoder3(out_encoder2)
        out_encoder4 = self.encoder4(out_encoder3)
        out_encoder5 = self.encoder5(out_encoder4)

        pose = self.pose_pred(out_encoder5)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        if self.output_exp:
            out_decoder5 = self.decoder5[1](torch.cat([self.upsample(self.decoder5[0](out_encoder5)), out_encoder4], dim=1))
            out_decoder4 = self.decoder4[1](torch.cat([self.upsample(self.decoder4[0](out_decoder5)), out_encoder3], dim=1))
            out_decoder3 = self.decoder3[1](torch.cat([self.upsample(self.decoder3[0](out_decoder4)), out_encoder2], dim=1))
            out_decoder2 = self.decoder2[1](torch.cat([self.upsample(self.decoder2[0](out_decoder3)), out_encoder1], dim=1))
            out_decoder1 = self.decoder1[1](self.upsample(self.decoder1[0](out_decoder2)))

            exp_mask4 = F.sigmoid(self.predict_mask4(out_decoder4))
            exp_mask3 = F.sigmoid(self.predict_mask3(out_decoder3))
            exp_mask2 = F.sigmoid(self.predict_mask2(out_decoder2))
            exp_mask1 = F.sigmoid(self.predict_mask1(out_decoder1))

        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
