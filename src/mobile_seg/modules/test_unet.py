from pathlib import Path
from turtle import forward

import torch
import torch.nn as nn

import torchvision.transforms.functional as TF
import torchvision.transforms as T

class Block (nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool =  nn.MaxPool2d(kernel_size = 2)

    def forward(self, x):
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features

class Decoder (nn.Module):
    def __init__(self, channels = None):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i],channels[i+1], kernel_size=2, stride=2) for i in range(len(channels)-1)])
        self.decoder_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.decoder_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        B,C,H,W = x.shape
        #enc_ftrs = T.CenterCrop([H,W])(enc_ftrs)
        enc_ftrs = T.Resize([H,W])(enc_ftrs)
        return enc_ftrs

class Unet(nn.Module):
    #def __init__(self, enc = (3, 64, 128, 256, 512, 1024), dec=(1024, 512, 256, 128, 64), num_class=11):
    def __init__(self, enc = (3, 16, 24, 32, 96, 320), dec=(160, 48, 16, 12), num_class=11):
        super().__init__()
        self.encoder = Encoder(enc)

    def forward(self, x):
        enc_ftrs = self.encoder(x)

        x = nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2)(enc_ftrs[4])
        x = torch.cat([x, enc_ftrs[3]], dim=1)
        x = nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)(x)

        x = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)(x)
        x = torch.cat([x, enc_ftrs[2]], dim=1)
        x = nn.Conv2d(80, 32, kernel_size=3, stride=1, padding=1)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)(x)

        x = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)(x)
        x = torch.cat([x, enc_ftrs[1]], dim=1)
        x = nn.Conv2d(40, 24, kernel_size=3, stride=1, padding=1)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)(x)

        x = nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2)(x)
        x = torch.cat([x, enc_ftrs[0]], dim=1)
        x = nn.Conv2d(28, 16, kernel_size=3, stride=1, padding=1)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)(x)

        x = nn.Conv2d(16, 11, 1)(x)

        return x

if __name__ == '__main__':

    x = torch.randn(1, 3, 512, 512)
    # encoder_block = Block(3,64)
    # print(encoder_block(x).shape)

    # encoder = Encoder()
    # ftrs = encoder(x)
    # print(f"Encoder outputs:")
    # for ftr in ftrs:
    #     print(ftr.shape)

    # decoder = Decoder()
    # y = torch.randn(1, 1024, 64, 64)
    # print(f"decoder output : {decoder(y, ftrs[::-1][1:]).shape}")

    unet = Unet()
    print(f"UNET output :{unet(x).shape}")

    
    
    

    