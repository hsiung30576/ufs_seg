# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import config
from base_model import resnet18
from seg_opr.seg_oprs import ConvBnRelu, LocationConfidence, AttentionFusion


def get():
    return conf(config.num_classes, None, None)


class conf(nn.Module):
    def __init__(self, out_planes, is_training,
                 criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(conf, self).__init__()
        
        self.is_training = is_training
        self.business_layer = []

        if is_training:
            self.criterion = criterion
        
        self.encoder = resnet18(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=config.bn_eps,
                                     bn_momentum=config.bn_momentum,
                                     deep_stem=False, stem_width=64)

        self.context_ff = AttentionFusion(256, 512, 128)
        self.spatial_conv = ConvBnRelu(64, 128, 1, 1, 0, dilation=1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.loc_conf = LocationConfidence(128+128, 1)

        self.refine_block = RefineOutput(128, out_planes, 4)
        self.spatial_refine_block = RefineOutput(128, out_planes, 4)
        self.context_refine_block = RefineOutput(128, out_planes, 16)

        self.business_layer.append(self.context_ff)
        self.business_layer.append(self.spatial_conv)
        self.business_layer.append(self.loc_conf)
        self.business_layer.append(self.refine_block)
        self.business_layer.append(self.spatial_refine_block)
        self.business_layer.append(self.context_refine_block)

    def forward(self, data, label=None):
        ori_features = self.encoder(data)
        #spatial
        spatial = ori_features[0]
        spatial = self.spatial_conv(spatial)
        spatial_output = self.spatial_refine_block(spatial)

        context_1 = ori_features[2]
        context_2 = F.interpolate(ori_features[3], scale_factor=2,
                                    mode='bilinear', align_corners=True)
        context = self.context_ff(context_1, context_2)
        context_output = self.context_refine_block(context)
        
        context = F.interpolate(context, scale_factor=4,
                                    mode='bilinear', align_corners=True)
        

        #location confidence
        lc = self.loc_conf(spatial, context)

        spatial = spatial * lc
        fcs = spatial + context

        output = self.refine_block(fcs)

        if self.is_training:
            loss = self.criterion(output, label)
            spatial_loss = self.criterion(spatial_output, label)
            context_loss = self.criterion(context_output, label)
            loss = loss + 0.4 * context_loss + 0.1*spatial_loss
            return loss

        return F.log_softmax(output, dim=1)

class RefineOutput(nn.Module):
    def __init__(self, in_planes, out_planes, scale=1, norm_layer=nn.BatchNorm2d):
        super(RefineOutput, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        return output