import time

import torch
import torch.nn as nn
import math

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from thop import profile
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary



class VSFE(nn.Module):
    def __init__(self, out_c, **kwargs):
        super(VSFE, self).__init__(**kwargs)
        self.branches = nn.ModuleList()
        for i in range(1, out_c + 1):
            branch = self.create_branch(i)
            self.branches.append(branch)
        self.batch_norm = nn.BatchNorm1d(out_c)
        self.channel_weights = nn.Parameter(torch.ones(out_c))
        self.pw1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1, padding=1)
        self.pw2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

    def forward(self, x):

        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        x1 = torch.cat(branch_outputs, dim=1)
        B, C, L = x1.size()
        x2 = x1 * self.channel_weights.view(1, C, 1)
        x3 = F.gelu(x2)
        x4 = self.batch_norm(x3)
        new_H = new_W = int(math.sqrt(L))
        x4 = x4.view(B, C, new_H, new_W)
        x4 = self.pw1(x4)
        x4 = self.pw2(x4)
        x4 = x4 * self.channel_weights.view(1, C, 1, 1)

        return x4

    def create_branch(self, i):
        branch = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=2 * i - 1, padding=(2 * i - 1 - 1) // 2, stride=1),
        )
        return branch