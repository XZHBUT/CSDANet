import time

import torch
import torch.nn as nn
import math
from module.VSFE import VSFE
from module.Attentionblock import WindowChannelAttention, MultibranchSpatialAttention
from module.Backbone import CSDAblock,BasicBlock
from module.Classifier import GroupedLinear

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from thop import profile
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary


class ResNet_CSDA_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = VSFE(8)
        # self.Encoder = nn.Sequential(
        #     BasicBlock(1, 8, 16, 32),
        #     BasicBlock(2, 16, 32, 16),
        #     BasicBlock(3, 32, 64, 8),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )
        self.BB1 = BasicBlock(1, 8, 16, 32)
        self.BB2 = BasicBlock(2, 16, 32, 16)
        self.BB3 = BasicBlock(3, 32, 64, 8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.Feedforward = nn.Sequential(
            GroupedLinear(64, 128, 16),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.lastLinear = nn.Linear(128, 10)
        nn.init.xavier_uniform_(self.Feedforward[0].weight)
        self.zero_last_layer_weight()

    def zero_last_layer_weight(self):
        self.lastLinear.weight.data = torch.zeros_like(self.lastLinear.weight)
        self.lastLinear.bias.data = torch.zeros_like(self.lastLinear.bias)

    def forward(self, data):
        FeatureHead = self.head(data)
        x, upx = self.BB1(FeatureHead)
        x, upx = self.BB2(x, upx)
        x = self.BB3(x, upx)
        x = self.pool(x)

        B, C, H, W = x.size()
        x = x.view(B, -1)
        x = self.Feedforward(x)
        out = self.lastLinear(x)

        return out




