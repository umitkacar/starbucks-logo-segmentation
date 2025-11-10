from timm.models.efficientnet import (
    efficientnet_lite0,
    mobilenetv2_035,
    mobilenetv2_050,
    mobilenetv2_075,
    mobilenetv2_100,
    mobilenetv2_120d,
    mobilenetv2_140,
)
from timm.models.efficientnet_builder import efficientnet_init_weights
import torch
from torch import nn

from mylib.pytorch_lightning.base_module import load_pretrained_dict, load_pretrained_dict_coreml


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False,
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ],
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class UpSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(UpSampleBlock, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.invres = InvertedResidual(out_channels * 2, out_channels, 1, 6)

    def forward(self, x0, x1):
        x = torch.cat([x0, self.dconv(x1)], dim=1)
        x = self.invres(x)
        return x


class MobileNetV2_unet(nn.Module):
    def __init__(self, arch_name=None, io_ratio=None, category=None, num_classes=None, **kwargs):
        super(MobileNetV2_unet, self).__init__()
        self.arch_name = arch_name
        self.io_ratio = io_ratio
        self.category = category
        self.num_classes = num_classes

        if self.arch_name in ["mobilenetv2_140", "mobilenetv2_120d"]:

            if self.arch_name == "mobilenetv2_140":
                self.backbone = mobilenetv2_140(pretrained=True, **kwargs)
                self.up_sample_blocks = nn.ModuleList(
                    [
                        UpSampleBlock(1792, 136),
                        UpSampleBlock(136, 48),
                        UpSampleBlock(48, 32),
                        UpSampleBlock(32, 24),
                    ],
                )

            if self.arch_name == "mobilenetv2_120d":
                self.backbone = mobilenetv2_120d(pretrained=True, **kwargs)
                self.up_sample_blocks = nn.ModuleList(
                    [
                        UpSampleBlock(1280, 112),
                        UpSampleBlock(112, 40),
                        UpSampleBlock(40, 32),
                        UpSampleBlock(32, 24),
                    ],
                )

            if self.category == "binary":

                if self.io_ratio == "half":

                    self.conv_last = nn.Sequential(
                        nn.Conv2d(24, 3, 1), nn.Conv2d(3, self.num_classes, 1), nn.Sigmoid(),
                    )

                elif self.io_ratio == "same":

                    self.conv_last = nn.Sequential(
                        nn.ConvTranspose2d(24, 24, 4, stride=2, padding=1, bias=False),
                        nn.Conv2d(24, self.num_classes, 1),
                        nn.Sigmoid(),
                    )

            if self.category == "multi":

                if self.io_ratio == "half":

                    self.conv_last = nn.Sequential(
                        nn.Conv2d(24, 24, 1),
                        nn.Conv2d(24, self.num_classes, 1),
                    )

                elif self.io_ratio == "same":

                    self.conv_last = nn.Sequential(
                        nn.ConvTranspose2d(24, 24, 4, stride=2, padding=1, bias=False),
                        nn.Conv2d(24, self.num_classes, 1),
                    )

        elif self.arch_name in ["mobilenetv2_100", "mobilenetv2_075", "efficientnet_lite0"]:

            if self.arch_name == "mobilenetv2_100":
                self.backbone = mobilenetv2_100(pretrained=True, **kwargs)
                self.up_sample_blocks = nn.ModuleList(
                    [
                        UpSampleBlock(1280, 96),
                        UpSampleBlock(96, 32),
                        UpSampleBlock(32, 24),
                        UpSampleBlock(24, 16),
                    ],
                )

            if self.arch_name == "mobilenetv2_075":
                self.backbone = mobilenetv2_075(pretrained=True, **kwargs)
                self.up_sample_blocks = nn.ModuleList(
                    [
                        UpSampleBlock(1280, 72),
                        UpSampleBlock(72, 24),
                        UpSampleBlock(24, 24),
                        UpSampleBlock(24, 16),
                    ],
                )

            if self.arch_name == "efficientnet_lite0":
                self.backbone = efficientnet_lite0(pretrained=True, **kwargs)
                self.up_sample_blocks = nn.ModuleList(
                    [
                        UpSampleBlock(1280, 112),
                        UpSampleBlock(112, 40),
                        UpSampleBlock(40, 24),
                        UpSampleBlock(24, 16),
                    ],
                )

            if self.category == "binary":

                if self.io_ratio == "half":

                    self.conv_last = nn.Sequential(
                        nn.Conv2d(16, 3, 1), nn.Conv2d(3, self.num_classes, 1), nn.Sigmoid(),
                    )

                elif self.io_ratio == "same":

                    self.conv_last = nn.Sequential(
                        nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
                        nn.Conv2d(16, self.num_classes, 1),
                        nn.Sigmoid(),
                    )

            if self.category == "multi":

                if self.io_ratio == "half":

                    self.conv_last = nn.Sequential(
                        nn.Conv2d(16, 16, 1),
                        nn.Conv2d(16, self.num_classes, 1),
                    )

                elif self.io_ratio == "same":

                    self.conv_last = nn.Sequential(
                        nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
                        nn.Conv2d(16, self.num_classes, 1),
                    )

        elif self.arch_name in ["mobilenetv2_050", "mobilenetv2_035"]:

            if self.arch_name == "mobilenetv2_050":
                self.backbone = mobilenetv2_050(pretrained=True, **kwargs)
                self.up_sample_blocks = nn.ModuleList(
                    [
                        UpSampleBlock(1280, 48),
                        UpSampleBlock(48, 16),
                        UpSampleBlock(16, 16),
                        UpSampleBlock(16, 8),
                    ],
                )
            if self.arch_name == "mobilenetv2_035":
                self.backbone = mobilenetv2_035(pretrained=True, **kwargs)
                self.up_sample_blocks = nn.ModuleList(
                    [
                        UpSampleBlock(1280, 32),
                        UpSampleBlock(32, 16),
                        UpSampleBlock(16, 8),
                        UpSampleBlock(8, 8),
                    ],
                )

            if self.category == "binary":

                if self.io_ratio == "half":

                    self.conv_last = nn.Sequential(
                        nn.Conv2d(8, 3, 1), nn.Conv2d(3, self.num_classes, 1), nn.Sigmoid(),
                    )

                elif self.io_ratio == "same":

                    self.conv_last = nn.Sequential(
                        nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1, bias=False),
                        nn.Conv2d(8, self.num_classes, 1),
                        nn.Sigmoid(),
                    )

            if self.category == "multi":

                if self.io_ratio == "half":

                    self.conv_last = nn.Sequential(
                        nn.Conv2d(8, 8, 1),
                        nn.Conv2d(8, self.num_classes, 1),
                    )

                elif self.io_ratio == "same":

                    self.conv_last = nn.Sequential(
                        nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1, bias=False),
                        nn.Conv2d(8, self.num_classes, 1),
                    )

        if self.arch_name == "mobileone":
            del self.backbone.gap, self.backbone.linear
        else:
            del (
                self.backbone.bn2,
                self.backbone.act2,
                self.backbone.global_pool,
                self.backbone.classifier,
            )

        efficientnet_init_weights(self.up_sample_blocks)
        efficientnet_init_weights(self.conv_last)

    def forward(self, x):

        down_feats = []
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)

        down_feats = []
        for b in self.backbone.blocks:

            x = b(x)
            if (self.arch_name == "efficientnet_lite0" and x.shape[1] in [16, 24, 40, 112]) or (self.arch_name == "mobilenetv2_120d" and x.shape[1] in [24, 32, 40, 112]) or (self.arch_name == "mobilenetv2_140" and x.shape[1] in [24, 32, 48, 136]) or (self.arch_name == "mobilenetv2_100" and x.shape[1] in [16, 24, 32, 96]) or (self.arch_name == "mobilenetv2_075" and x.shape[1] in [16, 24, 72]) or (self.arch_name == "mobilenetv2_050" and x.shape[1] in [8, 16, 48]) or (self.arch_name == "mobilenetv2_035" and x.shape[1] in [8, 16, 32]):
                down_feats.append(x)

        x = self.backbone.conv_head(x)

        for f, b in zip(reversed(down_feats), self.up_sample_blocks):
            x = b(f, x)

        x = self.conv_last(x)

        return x


def load_trained_model(config):

    if config["coreml"]:
        state_dict = load_pretrained_dict_coreml(config["ckpt_path"])
    else:
        state_dict = load_pretrained_dict(config["ckpt_path"])

    model = MobileNetV2_unet(
        arch_name=config["arch_name"],
        io_ratio=config["io_ratio"],
        category=config["category"],
        num_classes=config["num_classes"],
    )

    model.load_state_dict(state_dict)
    return model
