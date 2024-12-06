import torch
import torch.nn as nn
import numpy as np
import math

from torch.nn.parameter import Parameter
from torch.nn import functional as F

from typing import Union

#The ResNet models for CIFAR in https://arxiv.org/abs/1512.03385.

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

gn_groups = 4

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(gn_groups, planes, affine=False) 
        #self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(gn_groups, planes, affine=False) 
        #self.bn2 = nn.BatchNorm2d(planes, affine=False)


        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out) 

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, k=1, inchannel=3):
        super(ResNet, self).__init__()

        self.num_layers = sum(layers)
        self.inplanes = 16 * k
        self.conv1 = conv3x3(inchannel, 16 * k)
        self.gn1 = nn.GroupNorm(gn_groups, 16 * k, affine=False) 
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16 * k, layers[0])
        self.layer2 = self._make_layer(block, 32 * k, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * k, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * k, num_classes)

        # standard initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                try:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                except:
                    pass

        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.GroupNorm(gn_groups, self.inplanes, affine=False),#nn.BatchNorm2d(self.inplanes, affine=False), 
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20(k=1, inchannel=3):
    """Constructs a ResNet-20 model.

    """
    model = ResNet(BasicBlock, [3, 3, 3], k=k, inchannel=inchannel)
    return model