import time

import torch
import torch.nn as nn
import math



class WindowChannelAttention(nn.Module):
    def __init__(self, in_planes, gamma=2, b=1, ratio=16):
        super(WindowChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        t = int(abs((math.log(in_planes, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.avg_pool(x)
        max = self.max_pool(x)

        B, C, H, W = max.shape

        avg2 = avg.view(B, C, 1)
        max2 = max.view(B, C, 1)

        avg_out = self.conv(avg2.transpose(-1, -2))
        max_out = self.conv(max2.transpose(-1, -2))

        avg_out = avg_out.transpose(-1, -2).unsqueeze(-1)
        max_out = max_out.transpose(-1, -2).unsqueeze(-1)
        out = avg_out + max_out
        return self.sigmoid(out)


class MultibranchSpatialAttention(nn.Module):
    def __init__(self, HW, gamma=1.2, b=1):
        super(MultibranchSpatialAttention, self).__init__()
        t = int(abs((math.log(HW, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(1, k), stride=1, padding=(0, int(k / 2)))
        self.weight1 = nn.Parameter(torch.ones(1))
        self.conv2 = nn.Conv2d(2, 1, kernel_size=(1, k - 2), stride=1, padding=(0, int((k - 2) / 2)))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.conv3 = nn.Conv2d(2, 1, kernel_size=(k, 1), stride=1, padding=(int(k / 2), 0))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.conv4 = nn.Conv2d(2, 1, kernel_size=(k - 2, 1), stride=1, padding=(int((k - 2) / 2), 0))
        self.weight4 = nn.Parameter(torch.ones(1))
        self.pw = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x) * self.weight1 + self.conv2(x) * self.weight2
        x2 = self.conv3(x) * self.weight3 + self.conv4(x) * self.weight4
        x = torch.cat([x1, x2], dim=1)
        x = self.pw(x)
        return self.sigmoid(x)