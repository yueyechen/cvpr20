from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math
import torch
import torch.nn.functional as F

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
  def __init__(self, use_se, embedding_size, block, planes, layers, drop_out, se_reduction=16, use_triplet=False, use_rgb=True, use_depth=False):
    self.use_se = use_se
    self.inplanes = planes[0]
    super(ResNet18, self).__init__()
    if use_rgb:
      if use_depth:
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=7, stride=2, padding=3,
                              bias=False)
      else:
        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
                            bias=False)
    else:
      self.conv1 = nn.Conv2d(5, planes[0], kernel_size=7, stride=2, padding=3,
                            bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0])
    self.relu = nn.PReLU()
    self.use_triplet = use_triplet

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
    emd = y
    y = F.relu(y) 
    y = self.fc61(y)
    if self.use_triplet:
      return emd, y

    return y 
 

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
       # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class ResNet18_multi_output(nn.Module):
  def __init__(self, use_se, embedding_size, block, planes, layers, drop_out, se_reduction=16, use_triplet=False, feature_c=256, multi_output=True, add=False):
    self.use_se = use_se
    self.inplanes = planes[0]
    super(ResNet18_multi_output, self).__init__()
    self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0])
    self.relu = nn.PReLU()
    self.use_triplet = use_triplet
    self.multi_output = multi_output
    self.add = add

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
    self.layer1 = self._make_layer(block, planes[0], layers[0])
    self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
    self.seL2 = SELayer(planes[1], se_reduction)
    self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
    self.gap = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(p=drop_out)
    self.fc51 = nn.Linear(planes[2], embedding_size)
    self.fc61 = nn.Linear(embedding_size, 2)
    self.conv3_c = nn.Conv2d(planes[2], feature_c, kernel_size=1, stride=1)
    self.conv2_c = nn.Conv2d(planes[1], feature_c, kernel_size=1, stride=1)
    self.conv2_avoid_alias = nn.Conv2d(feature_c, feature_c, kernel_size=3, stride=1, padding=1)
    self.gap2 = nn.AdaptiveAvgPool2d(1)
    self.fc2_emd = nn.Linear(feature_c, embedding_size)
    self.fc2_out = nn.Linear(embedding_size, 2)
    self.conv1_c = nn.Conv2d(planes[0], feature_c, kernel_size=1, stride=1)
    self.conv1_avoid_alias = nn.Conv2d(feature_c, feature_c, kernel_size=3, stride=1, padding=1)
    self.gap1 = nn.AdaptiveAvgPool2d(1)
    self.fc1_emd = nn.Linear(feature_c, embedding_size)
    self.fc1_out = nn.Linear(embedding_size, 2)
    if not self.multi_output:
      self.fc_concat = nn.Linear(feature_c * 3, embedding_size)
      self.fc_add = nn.Linear(feature_c, embedding_size)

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
    y1 = self.layer1(y)
    y2 = self.layer2(y1)
    if self.use_se:
      y2 = self.seL2(y2)
    y3 = self.layer3(y2)
    if not self.multi_output:
      p3 = self.conv3_c(y3)
      p2 = self.conv2_c(y2)
      p1 = self.conv1_c(y1)
      up_shape = p1.shape
      up3 = nn.Upsample((up_shape[-2], up_shape[-1]))
      up2 = nn.Upsample((up_shape[-2], up_shape[-1]))
      p3 = up3(p3)
      p2 = up2(p3)
      if self.add:
        y = self.gap(p1 + p2 + p3)
      else:
        y = self.gap(torch.cat((p1, p2, p3), dim=1))
      # y = self.gap(p1 + p2 + p3)
      y = self.dropout(y)
      y = y.view(y.size(0), -1)
      if self.add:
        y = self.fc_add(y)
      else:
        y = self.fc_concat(y)
      y = self.fc61(y) 
      return y
    else:
      y = self.gap(y3)
      y = self.dropout(y)
      y = y.view(y.size(0), -1)
      y = F.relu(self.fc51(y))
      y = self.fc61(y) 
      y3_up_sample_shape = y2.shape
      p3 = self.conv3_c(y3)
      up_sample_3_2 = nn.Upsample((y3_up_sample_shape[-2], y3_up_sample_shape[-1]))
      p3_up_sample = up_sample_3_2(p3)
      c2 = self.conv2_c(y2)
      p2 = c2 + p3_up_sample
      p2 = self.conv2_avoid_alias(p2)
      p2_fc = self.gap2(p2)
      p2_fc = p2_fc.view(p2_fc.size(0), -1)
      f2 = F.relu(self.fc2_emd(p2_fc))
      f2 = self.fc2_out(f2)
      p2_up_sample_shape = y1.shape
      up_sample_2_1 = nn.Upsample((p2_up_sample_shape[-2], p2_up_sample_shape[-1]))
      p2_up_sample = up_sample_2_1(p2)
      c1 = self.conv1_c(y1)
      p1 = c1 + p2_up_sample
      p1 = self.conv1_avoid_alias(p1)
      p1_fc = self.gap1(p1)
      p1_fc = p1_fc.view(p1_fc.size(0), -1)
      f1 = F.relu(self.fc1_emd(p1_fc))
      f1 = self.fc1_out(f1)
      return (f1 + f2 + y) / 3

def officali_resnet18(**kwargs):
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  return model

def resnet18(use_se=True, embedding_size=1024, drop_out = 0.7, se_reduction = 16, use_triplet = False, use_rgb = True, use_depth=False):
  model = ResNet18(use_se, embedding_size, BasicBlock,[64,128,256],[1,1,1], drop_out, se_reduction, use_triplet, use_rgb, use_depth)
  return model

def resnet18_concat(use_se=True, embedding_size=1024, drop_out = 0.7, se_reduction = 16, use_triplet = False, feature_c=256, multi_output=True, add=True):
  model = ResNet18_multi_output(use_se, embedding_size, BasicBlock,[64,128,256],[1,1,1], drop_out, se_reduction, use_triplet, feature_c, multi_output, add)
  return model
