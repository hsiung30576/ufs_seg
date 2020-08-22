# encoding: utf-8
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y.shape)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

class AttentionFusion(nn.Module):
    def __init__(self, low_in_planes, high_in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(AttentionFusion, self).__init__()
        self.channel_attention_1 = SELayer(high_in_planes, out_planes)
        self.channel_attention_2 = SELayer(high_in_planes, out_planes)
        self.conv_1x1_low = ConvBnRelu(low_in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1_high = ConvBnRelu(high_in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
    def forward(self, x1, x2):
        vec_1 = self.channel_attention_1(x2)
        vec_2 = self.channel_attention_2(x2)
        x1 = self.conv_1x1_low(x1)
        x2 = self.conv_1x1_high(x2)
        fm = x1 * vec_1 + x2 * vec_2

        return fm

class LocationConfidence(nn.Module):
    def __init__(self, in_planes, out_planes=1,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(LocationConfidence, self).__init__()
        inner_channel = 64
        self.conv_1x1 = ConvBnRelu(in_planes, inner_channel, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.Confidence = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(inner_channel, inner_channel, 3, 1, 1,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_c = self.Confidence(fm)
        return fm_c
