from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
import torchvision.models as models


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def l2_norm(in_data, axis=1):
    norm = torch.norm(in_data, 2, axis, True)
    output = torch.div(in_data, norm)

    return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),

            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet18(nn.Module):
    def __init__(self, in_plane, use_se, embedding_size, block,
                 planes, layers, drop_out, se_reduction=16, num_classes=2):
        self.use_se = use_se
        self.inplanes = planes[0]
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, planes[0], kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu = nn.PReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.seL2 = SELayer(planes[1], se_reduction)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_out)
        self.fc51 = nn.Linear(planes[2], embedding_size)
        self.fc61 = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        if self.use_se:
            y = self.seL2(y)
        y = self.layer3(y)
        y = self.gap(y)
        y = self.dropout(y)
        y = y.view(y.size(0), -1)
        y = self.fc51(y)
        output = self.fc61(y)

        return y, output


def resnet18(in_plane, use_senet=True, embedding_size=1024,
             drop_out=0.7, se_reduction=16, num_classes=2):
    model = ResNet18(in_plane=in_plane,
                     use_se=use_senet,
                     embedding_size=embedding_size,
                     block=BasicBlock,
                     planes=[32, 64, 128, 256], layers=[2, 2, 2, 2],
                     drop_out=drop_out,
                     se_reduction=se_reduction,
                     num_classes=num_classes)

    return model
