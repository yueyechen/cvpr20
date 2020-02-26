from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1): # def conv3x3(in_planes, out_planes, stride=1): #
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

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
  def __init__(self, use_se, embedding_size, block, planes, layers, drop_out, se_reduction=16):
    self.use_se = use_se
    self.inplanes = planes[0]
    super(ResNet18, self).__init__()
    # self.conv1 = nn.Conv2d(5, planes[0], kernel_size=7, stride=2, padding=3,
    #                        bias=False)
    self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0])
    self.relu = nn.PReLU()

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
    self.layer1 = self._make_layer(block, planes[0], layers[0])
    self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
    self.seL2 = SELayer(planes[1], se_reduction)
    self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
    self.gap = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(p=drop_out)
    self.fc51 = nn.Linear(planes[2], embedding_size)
    self.fc61 = nn.Linear(embedding_size, 2)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

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
    y = self.fc61(y)

    return y

def resnet18(use_se=True, embedding_size=1024, drop_out = 0.7, se_reduction = 16):
  model = ResNet18(use_se, embedding_size, BasicBlock,[64,128,256],[1,1,1], drop_out, se_reduction)
  return model
