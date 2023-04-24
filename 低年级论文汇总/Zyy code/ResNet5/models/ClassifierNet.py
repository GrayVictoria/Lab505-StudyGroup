import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FirstResBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(FirstResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x
        out = F.relu(out)
        return out

class Classifier(nn.Module):
    def __init__(self, hidden_size, patch_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(3 * hidden_size * patch_size * patch_size, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out

class Net(nn.Module):
    def __init__(self, hsi_channels, sar_channels, dsm_channels, hidden_size, patch_size, num_classes):
        self.planes = hidden_size

        super(Net, self).__init__()
        self.res_for_hsi = FirstResBlock(hsi_channels, hidden_size)
        self.res_for_sar = FirstResBlock(sar_channels, hidden_size)
        self.res_for_dsm = FirstResBlock(dsm_channels, hidden_size)

        self.res_blocks = nn.ModuleList([ResBlock(hidden_size) for _ in range(12)])

        self.classifier = Classifier(hidden_size, patch_size, num_classes)

    def forward(self, x, y, z):
        x = self.res_for_hsi(x)
        y = self.res_for_sar(y)
        z = self.res_for_dsm(z)
        for i in range(4):
            x = self.res_blocks[i * 3](x)
            y = self.res_blocks[i * 3 + 1](y)
            z = self.res_blocks[i * 3 + 2](z)

        out = torch.cat((x, y, z), 1)
        out = self.classifier(out)
        return out
