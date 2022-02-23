#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *
from utils import PreEmphasis

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
    
class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=448, log_input=True, **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        # self.conv1 = nn.Conv2d(3, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        # self.pooling = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm   = nn.InstanceNorm12d(n_mels)  # c = 3
        outmap_size = int(self.n_mels/8)
        
        #self.avg_pool = GeM()
        
        self.attention = nn.Sequential(
            nn.Conv2d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=-1),
            )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
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

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):

        x = self.instancenorm(x) # B 3 448 448

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = x.reshape(x.size()[0],-1,x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=(2, 3))
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=(2, 3))
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=(2, 3)) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg),1)  # 28 * 28 = 784

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model

