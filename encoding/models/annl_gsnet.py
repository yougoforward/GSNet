from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['annl_gsnetNet', 'get_annl_gsnetnet']


class annl_gsnetNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(annl_gsnetNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = annl_gsnetNetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        x = list(self.head(c4))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class annl_gsnetNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(annl_gsnetNetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.aa_annl_gsnet = annl_gsnet_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1))
        if self.se_loss:
            self.selayer = nn.Linear(inter_channels, out_channels)

    def forward(self, x):
        feat_sum, gap_feat = self.aa_annl_gsnet(x)
        outputs = [self.conv8(feat_sum)]
        if self.se_loss:
            outputs.append(self.selayer(torch.squeeze(gap_feat)))

        return tuple(outputs)


def annl_gsnetConv(in_channels, out_channels, atrous_rate, norm_layer):
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


class annl_gsnetPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(annl_gsnetPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

        self.out_chs = out_channels

    def forward(self, x):
        bs, _, h, w = x.size()
        pool = self.gap(x)

        # return F.interpolate(pool, (h, w), **self._up_kwargs)
        # return pool.repeat(1,1,h,w)
        return pool.expand(bs, self.out_chs, h, w)


class annl_gsnet_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(annl_gsnet_Module, self).__init__()
        # out_channels = in_channels // 4
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = annl_gsnetConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = annl_gsnetConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = annl_gsnetConv(in_channels, out_channels, rate3, norm_layer)

        self._up_kwargs = up_kwargs
        self.psaa_conv = nn.Sequential(nn.Conv2d(in_channels+4*out_channels, out_channels, 1, padding=0, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(out_channels, 4, 1, bias=True),
                                    nn.Sigmoid())  

        self.project = nn.Sequential(nn.Conv2d(in_channels=4*out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(out_channels),
                      nn.ReLU(True))


        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, out_channels, 1, bias=False),
                            norm_layer(out_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, 1, bias=True),
                            nn.Sigmoid())

        self.pam0 = APAM_Module(in_dim=out_channels, key_dim=out_channels//2,value_dim=out_channels//2,out_dim=out_channels,norm_layer=norm_layer)
    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        n, c, h, w = feat0.size()

        # psaa
        y1 = torch.cat((x, feat0, feat1, feat2, feat3), 1)
        psaa_att = self.psaa_conv(y1)
        psaa_att_list = torch.split(psaa_att, 1, dim=1)

        y2 = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2,
                        psaa_att_list[3] * feat3), 1)
        out = self.project(y2)
        
        #gp
        gp = self.gap(x)
        
        # se
        se = self.se(gp)
        out = out + se*out

        #non-local
        out = self.pam0(out)

        out = torch.cat([out, gp.expand(n, c, h, w)], dim=1)
        return out, gp

def get_annl_gsnetnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = annl_gsnetNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

# class APAM_Module(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer, psp_size=(1,3,6,8)):
#         super(APAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.key_channels = key_dim
#         self.psp = PSPModule(psp_size)
#         self.query_conv = nn.Sequential(
#             nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1, bias=False),
#             norm_layer(key_dim),
#             nn.ReLU(True))
#         self.key_conv = self.query_conv
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=value_dim, kernel_size=1)
#         self.gamma = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, bias=True), nn.Sigmoid())

#         self.softmax = nn.Softmax(dim=-1)
#         self.fuse_conv = nn.Sequential(nn.Conv2d(value_dim, out_dim, 1, bias=False),
#                                        norm_layer(out_dim),
#                                        nn.ReLU(True))

#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.psp(self.key_conv(x))
#         energy = torch.bmm(proj_query, proj_key)

#         energy = (self.key_channels ** -.5) * energy
#         attention = self.softmax(energy)
#         proj_value = self.psp(self.value_conv(x))
        
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, -1, height, width)
        
#         out =self.fuse_conv(out)
#         gamma = self.gamma(x)
#         out = (1-gamma)*out + gamma*x
#         return out

class APAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer, psp_size=(1,3,6,8)):
        super(APAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.psp = PSPModule(psp_size)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.gamma = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, bias=True), nn.Sigmoid())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        xp = self.psp(x)
        m_batchsize, C, height, width = x.size()
        m_batchsize, C, hpwp = xp.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(xp)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = xp
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        gamma = self.gamma(x)
        out = (1-gamma)*out + gamma*x
        return out