from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['CAN', 'get_can']

class CAN(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, atrous_rates=(12, 24, 36), norm_layer=nn.BatchNorm2d, **kwargs):
        super(CAN, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = CANHead(2048, nclass, norm_layer, self._up_kwargs,atrous_rates)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)

        outputs = []
        x, xe = self.head(c4, c1)
        x = F.interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        xe = F.interpolate(xe, (h,w), **self._up_kwargs)
        outputs.append(xe)
        
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)


class CANHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates):
        super(CANHead, self).__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        # self.block = nn.Sequential(
        #     nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #     norm_layer(inter_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.1, False),
        #     nn.Conv2d(inter_channels, out_channels, 1))
        self.block = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

        self.skip = nn.Sequential(
            nn.Conv2d(inter_channels, 48, 1, padding=0, bias=False),
            norm_layer(48),
            nn.ReLU(True)
            )

        self.decoder = nn.Sequential(
            nn.Conv2d(inter_channels+48, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
            )
        self._up_kwargs = up_kwargs
        
        self.cam = CAM_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        self.se = nn.Sequential(
                            nn.Conv2d(inter_channels, out_channels, 1, bias=True),
                            nn.Sigmoid())
        self.block2 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))
    def forward(self, x, xl):
        n,c,h,w = xl.size()
        x_aspp = self.aspp(x)
        xup = F.interpolate(x_aspp, (h,w), **self._up_kwargs)
        x_skip = self.skip(xl)
        out = self.decoder(torch.cat([xup, x_skip], dim=1))
        out = self.block(out)

        #class-aware attention
        xe = self.cam(x)
        xe_up = F.interpolate(xe, (h,w), **self._up_kwargs)
        out = self.se(xe_up)*out
        return out, self.block2(xe)

# def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
#     block = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
#                   dilation=atrous_rate, bias=False),
#         norm_layer(out_channels),
#         nn.ReLU(True))
#     return block

def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, 512, 1, padding=0,
                  dilation=1, bias=False),
        norm_layer(512),
        nn.ReLU(True),
        nn.Conv2d(512, out_channels, 3, padding=atrous_rate,
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

class CAM_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(CAM_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        # self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        # self.project = nn.Sequential(
        #     nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.5, False))
        self.project = nn.Sequential(
            nn.Conv2d(4*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, out_channels, 1, bias=False),
                            norm_layer(out_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, 1, bias=True),
                            nn.Sigmoid())

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        # feat4 = self.b4(x)
        gp = self.gap(x)
        se = self.se(gp)

        y = torch.cat((feat0, feat1, feat2, feat3), 1)
        y = self.project(y)

        return y+se*y
def get_can(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = CAN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
