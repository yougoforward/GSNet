from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate, unfold

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['up_deeplabv3', 'get_up_deeplabv3']

class up_deeplabv3(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(up_deeplabv3, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = up_deeplabv3Head(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c1,c2,c4)
        x = F.interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)


class up_deeplabv3Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates=(12, 24, 36)):
        super(up_deeplabv3Head, self).__init__()
        inter_channels = in_channels // 8
        self.conv5 = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        # self.block = nn.Sequential(
        #     nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #     norm_layer(inter_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.1, False),
        #     nn.Conv2d(inter_channels, out_channels, 1))
        self.conv6 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

        self.refine = nn.Sequential(nn.Conv2d(256, 64, 3, padding=2, dilation=2, bias=False),
                                   norm_layer(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
                                   norm_layer(64),
                                   nn.ReLU())
        self.refine2 = nn.Sequential(nn.Conv2d(512, 64, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(64),
                                   nn.ReLU()) 
        self._up_kwargs = up_kwargs

    def forward(self, c1,c2,x):
        n,c,h,w =c1.size()
        c1 = self.refine(c1) # n, 64, h, w
        c2 = interpolate(c2, (h,w), **self._up_kwargs)
        c2 = self.refine2(c2)

        unfold_up_c2 = unfold(c2, 3, 2, 2, 1).view(n, 64, 3*3, h*w)
        # torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
        energy = torch.matmul(c1.view(n, 64, 1, h*w).permute(0,3,2,1), unfold_up_c2.permute(0,3,1,2)) #n,h*w,1,3x3
        att = torch.softmax(energy, dim=-1)
        out =self.conv5(x)
        out0 = interpolate(out, (h,w), **self._up_kwargs)
        unfold_out = unfold(out0, 3, 2, 2, 1).view(n, 512, 3*3, h*w)
        out = torch.matmul(att, unfold_out.permute(0,3,2,1)).permute(0,3,2,1).view(n,512,h,w)
        return self.conv6(out)


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h,w), **self._up_kwargs)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        # self.project = nn.Sequential(
        #     nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.5, False))
        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)

        return self.project(y)


def get_up_deeplabv3(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = up_deeplabv3(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
