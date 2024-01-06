import torch
import torch.nn as nn
import math
from VSFE import VSFE
from Attentionblock import WindowChannelAttention, MultibranchSpatialAttention

class LayerScale_Sa(nn.Module):
    def __init__(self, hidden_size, init_ones=True):
        super().__init__()
        if init_ones:
            self.alpha = nn.Parameter(torch.ones(hidden_size) * 0.1)
        else:
            self.alpha = nn.Parameter(torch.zeros(hidden_size))
        self.move = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):

        out = x * self.alpha + self.move


        return out


class LayerScale_Ca(nn.Module):
    def __init__(self, channel, init_ones=True):
        super().__init__()
        if init_ones:
            self.alpha = nn.Parameter(torch.ones(channel) * 0.1)
        else:
            self.alpha = nn.Parameter(torch.zeros(channel))
        self.move = nn.Parameter(torch.zeros(channel))

    def forward(self, x):


        out = x * self.alpha.view(1, -1, 1, 1) + self.move.view(1, -1, 1, 1)

        return out



class CSDAblock(nn.Module):
    def __init__(self, numb, in_planes, HW, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ca = WindowChannelAttention(in_planes)
        self.sa = MultibranchSpatialAttention(HW)
        self.CaWeight = LayerScale_Ca(in_planes)
        self.SaWeight = LayerScale_Sa(HW)
        self.PW_Proj = nn.Conv2d(2 * in_planes, in_planes, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.numb = numb

    def forward(self, x, upx=None):
        CaSaOut = self.CaWeight(self.ca(x)) * self.SaWeight(self.sa(x))
        if self.numb == 1:
            breach_x = x
            out = torch.cat([CaSaOut + breach_x, breach_x], dim=1)
            out = self.PW_Proj(out)
            return self.relu(self.bn(out)), breach_x
        elif self.numb == 2 or 3:
            breach_x = x + upx
            out = torch.cat([CaSaOut + breach_x, breach_x], dim=1)
            out = self.PW_Proj(out)
            if self.numb == 3:
                return self.relu(self.bn(out))
            return self.relu(self.bn(out)), breach_x


class BasicBlock(nn.Module):
    def __init__(self, numb, in_channel, out_channel, HW, stride=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numb = numb
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.CBAM = CSDAblock(numb, in_channel, HW)
        self.PW_Proj = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.PW_Proj2 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, upx=None):
        if self.numb == 1:
            CBAMOut, upx2 = self.CBAM(x, upx)
            out = CBAMOut + x
            out = self.PW_Proj(out)
            out = self.pooling(out)
            breach_out = self.pooling(self.PW_Proj2(upx2))
            return self.relu(self.bn(out)), breach_out
        elif self.numb == 2:
            CBAMOut, upx2 = self.CBAM(x, upx)
            out = CBAMOut + x
            out = self.PW_Proj(out)
            out = self.pooling(out)
            breach_out = self.pooling(self.PW_Proj2(upx2))
            return self.relu(self.bn(out)), breach_out
        else:
            CBAMOut = self.CBAM(x, upx)
            out = CBAMOut + x
            out = self.PW_Proj(out)
            out = self.pooling(out)
            return self.relu(self.bn(out))