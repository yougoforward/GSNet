###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import interpolate, unfold

from .base import BaseNet
from .fcn import FCNHead
from ..nn import PyramidPooling

class up_psp(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(up_psp, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = up_pspHead(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c1,c2,c4)
        x = interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class up_pspHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(up_pspHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
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

def get_up_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = up_psp(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('up_psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_up_psp_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""up_psp model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_up_psp('ade20k', 'resnet50', pretrained, root=root, **kwargs)
