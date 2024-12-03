import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.mutilsparetrnasformer import TransformerBlock
from models.hilo import Block

# source from https://github.com/Yejin0111/ADD-GCN


class DGG(nn.Module):
    def __init__(self, input_channels, num_nodes, num_classes,  patch_size, drop_prob=0.1):
        super(DGG, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.mutilspare1=TransformerBlock(patch_size=1, dim=64, num_heads=num_nodes, ffn_expansion_factor=2, stride=1)
        self.mutilspare2 = TransformerBlock(patch_size=1, dim=64, num_heads=num_nodes, ffn_expansion_factor=2, stride=1)
        self.pu1=Block(64, num_nodes)
        self.pul2=Block(64,num_nodes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))

        self.fc1 = nn.Linear(256, 64)
        self.bn_f1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.squeeze()


        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x1=self.mutilspare1(x)
        x1=self.mutilspare2(x1)#256.64.3.3

        x_3 = self.avgpool(x1)
        x_4 = self.maxpool(x1)

        x_3 = x_3.view(-1,x_3.size(1) * x_3.size(2) * x_3.size(3))
        x_4 = x_4.view(-1, x_4.size(1) * x_4.size(2) * x_4.size(3))
        x_5 = torch.cat((x_3, x_4), dim=-1)

        x2 = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=True)#256.64.18.18
        x2=self.pu1(x2.permute(0,2,3,1))
        x2=self.pul2(x2.permute(0,2,3,1))
        x_9 = self.avgpool1(x2)
        x_10 = self.maxpool1(x2)

        x_9 = x_9.view(-1, x_9.size(1) * x_9.size(2) * x_9.size(3))
        x_10 = x_10.view(-1, x_10.size(1) * x_10.size(2) * x_10.size(3))
        x = torch.cat((x_9, x_10,x_5), dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = self.bn_f1(x)

        x = F.leaky_relu(self.fc2(x))

        return x,x









