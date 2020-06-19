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

__all__ = ['up_fcn', 'get_up_fcn', 'get_up_fcn_resnet50_pcontext', 'get_up_fcn_resnet50_ade']

class up_fcn(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = up_fcn(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(up_fcn, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = up_fcnHead(2048, nclass, norm_layer)
        if aux:
            self.auxlayer = up_fcnHead(1024, nclass, norm_layer)


    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)
        x = self.head(c1,c2,c4)

        x = interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

        
class up_fcnHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(up_fcnHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )

        self.refine = nn.Sequential(nn.Conv2d(256, 64, 3, padding=2, dilation=2, bias=False),
                                   norm_layer(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
                                   norm_layer(64),
                                   nn.ReLU())
        self.refine2 = nn.Sequential(nn.Conv2d(512, 64, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(64),
                                   nn.ReLU()) 

        self.refine3 = nn.Sequential(nn.Conv2d(512, 64, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(64),
                                   nn.ReLU()) 
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, c1,c2,x):
        n,c,h,w =c1.size()
        c1 = self.refine(c1) # n, 64, h, w
        c2 = interpolate(c2, (h,w), **self._up_kwargs)
        up_c2 = self.refine2(c2)

        unfold_up_c2 = unfold(up_c2, 3, 2, 2, 1).view(n, 64, 3*3, h*w)
        # torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
        energy = torch.matmul(c1.view(n, 64, 1, h*w).permute(0,3,2,1), unfold_up_c2.permute(0,3,1,2)) #n,h*w,1,3x3
        att = torch.softmax(energy, dim=-1)
        out =self.conv5(x)
        out = interpolate(out, (h,w), **self._up_kwargs)
        unfold_out = unfold(out, 3, 2, 2, 1).view(n, 512, 3*3, h*w)
        out = torch.matmul(unfold_out.permute(0,3,1,2), att.permute(0,1,3,2)).permute(0,2,1,3).view(n,512,h,w)

        return self.conv6(out)


def get_up_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""up_fcn model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_up_fcn.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_up_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = up_fcn(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('up_fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_up_fcn_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_fcn_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_up_fcn('pcontext', 'resnet50', pretrained, root=root, aux=False, **kwargs)

def get_up_fcn_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_fcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_up_fcn('ade20k', 'resnet50', pretrained, root=root, **kwargs)
