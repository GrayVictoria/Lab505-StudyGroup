import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        out = out * x + x
        return out

class CIM(nn.Module):
    def __init__(self, hidden_size):
        super(CIM, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, 128, 1)
        self.conv2 = nn.Conv2d(hidden_size, 128, 1)
        self.ca = ChannelAttention(128)

    def forward(self, x, y, z):
        s1 = x + y
        s1 = self.conv1(s1)
        s1 = self.ca(s1)

        s2 = self.conv2(z)
        return s1 + s2

class Classifier(nn.Module):
    def __init__(self, hidden_size, patch_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_size * patch_size * patch_size, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class Net(nn.Module):
    def __init__(self, hsi_channels, sar_channels, hidden_size, patch_size, num_classes=7):
        self.planes = hidden_size
        self.num_parallel = 3

        super(Net, self).__init__()

        self.conv_x = nn.Conv2d(hsi_channels, hidden_size, 1, bias=False)
        self.conv_y = nn.Conv2d(sar_channels, hidden_size, 1, bias=False)
        self.conv_z = nn.Conv2d(sar_channels, hidden_size, 1, bias=False)
        self.bnx = nn.BatchNorm2d(hidden_size)
        self.bny = nn.BatchNorm2d(hidden_size)
        self.bnz = nn.BatchNorm2d(hidden_size)

        self.cim = nn.ModuleList([CIM(self.planes) for _ in range(5)])

        self.conv_out = nn.ModuleList([nn.Conv2d(hidden_size, hidden_size, 1, bias=False) for _ in range(4)])
        self.conv_sar1 = nn.ModuleList([nn.Conv2d(sar_channels, hidden_size, 1, bias=False) for _ in range(4)])
        self.conv_sar2 = nn.ModuleList([nn.Conv2d(sar_channels, hidden_size, 1, bias=False) for _ in range(4)])
        self.bn_out = nn.ModuleList([nn.BatchNorm2d(hidden_size) for _ in range(4)])
        self.bn_sar1 = nn.ModuleList([nn.BatchNorm2d(hidden_size) for _ in range(4)])
        self.bn_sar2 = nn.ModuleList([nn.BatchNorm2d(hidden_size) for _ in range(4)])

        self.classifier = Classifier(hidden_size, patch_size, num_classes)

    def forward(self, x, y, z):
        sar1 = y
        sar2 = z

        x = self.conv_x(x)
        x = self.bnx(x)
        x = F.relu(x)
        y = self.conv_y(y)
        y = self.bny(y)
        y = F.relu(y)
        z = self.conv_z(z)
        z = self.bnz(z)
        z = F.relu(z)
        out = self.cim[0](x, y, z)
        for i in range(4):
            out = self.conv_out[i](out)
            out = self.bn_out[i](out)
            out = F.relu(out)
            y = self.conv_sar1[i](sar1)
            y = self.bn_sar1[i](y)
            y = F.relu(y)
            z = self.conv_sar2[i](sar2)
            z = self.bn_sar2[i](z)
            z = F.relu(z)
            out = self.cim[i + 1](out, y, z)
        out = self.classifier(out)
        return out
