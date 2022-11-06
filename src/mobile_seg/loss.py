from torch.nn.functional import interpolate
import torch.nn as nn
import torch

def dice_loss(scale=None):
    
    def fn(input, target):
        smooth = 1.
        if scale == 2:
            input = interpolate(input, scale_factor=scale, mode='nearest')

        iflat = input.view(-1)
        iflat = iflat.float()
        tflat = target.view(-1).float()
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    return fn

def ce_loss(scale=2):

    def fn(input, target):
        if scale == 2:
            input = interpolate(input, scale_factor=scale, mode='nearest', align_corners=False)
        class_weights = torch.tensor([0.05, 0.01, 0.12, 0.05, 0.12, 0.12, 0.12, 0.05, 0.12, 0.12, 0.12]).to("cuda")
        return nn.CrossEntropyLoss(weight=class_weights,reduction='mean')(input, target)

    return fn
