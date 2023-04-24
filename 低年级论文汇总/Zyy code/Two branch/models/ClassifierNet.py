import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class cascade_block(nn.Module):
  def __init__(self, channel):
    super(cascade_block, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.conv1_1 = nn.Conv2d(64, channel*2, 3, padding=1)  # YC
    self.bn1_1 = nn.BatchNorm2d(channel*2)

    self.conv1_2 = nn.Conv2d(channel*2, channel, 1,)  # YC
    self.bn1_2 = nn.BatchNorm2d(channel)

    self.conv = nn.Conv2d(64, channel*2, 1,)

    self.conv2_1 = nn.Conv2d(channel, channel*2, 3, padding=1)  # YC
    self.bn2_1 = nn.BatchNorm2d(channel*2)

    self.conv2_2 = nn.Conv2d(channel*2, channel, 3, padding=1)  # YC
    self.bn2_2 = nn.BatchNorm2d(channel)

  def forward(self, x):

    conv1_1 = F.relu(self.bn1_1(self.conv1_1(x)))
    conv1_2 = F.relu(self.bn1_2(self.conv1_2(conv1_1)))
    x_1 = self.conv(x)
    conv2_1 = self.conv2_1(conv1_2)
    conv2_1 = x_1+conv2_1
    conv2_1 = F.relu(self.bn2_1(conv2_1))
    conv2_2 = self.bn2_2(self.conv2_2(conv2_1))
    conv2_2 = torch.add(conv1_2, conv2_2)
    conv2_2 = F.relu(conv2_2)

    return conv2_2


class cascadeNet(nn.Module):
  def __init__(self, sar_channels):
    super(cascadeNet, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.conv0 = nn.Conv2d(sar_channels, 64, 3, padding=1)  # YC
    self.cascade_block1 = cascade_block(64)

    self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    self.cascade_block2 = cascade_block(128)

    self.conv2 = nn.Conv2d(256, 128, 3)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 128, 3)
    self.bn3 = nn.BatchNorm2d(128)
    self.pool = nn.AdaptiveAvgPool2d(1)

    #self.fc1 = nn.Linear(3*3*128,128)
    #self.fc2 = nn.Linear(100,512)

  def forward(self, x):

    x = F.relu(self.conv0(x))
    x = self.cascade_block1(x)
    x = F.relu(self.max_pool(x))
    x = self.cascade_block2(x)
    x = self.pool(x)
    x1 = x.contiguous().view(x.size(0), -1)

    return x1


class pixel_branch(nn.Module):
  def __init__(self):
    super(pixel_branch, self).__init__()

    self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11)
    self.bn1 = nn.BatchNorm1d(64)

    self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)

    self.max_pool = nn.MaxPool1d(kernel_size=2)

    #self.fc1 = nn.Linear(3*3*128,128)
    #self.fc2 = nn.Linear(100,512)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = x.unsqueeze(1)
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = F.relu(self.conv2(x))
    x = self.max_pool(x)
    x1 = x.contiguous().view(x.size(0), -1)

    return x1


class simple_cnn_branch(nn.Module):
  def __init__(self, hsi_channels):
    super(simple_cnn_branch, self).__init__()

    self.conv1 = nn.Conv2d(hsi_channels, 256, 3, padding=1)  # YC
    self.bn1 = nn.BatchNorm2d(256)

    self.conv2 = nn.Conv2d(256, 128, 1)
    self.bn2 = nn.BatchNorm2d(128)
    self.max_pool = nn.MaxPool2d(kernel_size=2)

  def forward(self, x):

    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.max_pool(x)
    x1 = x.contiguous().view(x.size(0), -1)
    return x1


class lidar_branch(nn.Module):
  def __init__(self, sar_channels):
    super(lidar_branch, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.net = cascadeNet(sar_channels)

    self.dropout = nn.Dropout(p=0.5)
    self.fc = nn.Linear(128, 128)

  def forward(self, x):

    x = self.net(x)
    x = self.dropout(x)
    x = self.fc(x)

    return x


class hsi_branch(nn.Module):
  def __init__(self, hsi_channels, patch_size):
    super(hsi_branch, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.net1 = simple_cnn_branch(hsi_channels)
    self.net2 = pixel_branch()

    self.dropout = nn.Dropout(p=0.5)
    self.fc = nn.Linear((hsi_channels - 12) * 16 + 128 * (patch_size // 2) * (patch_size // 2), 128)

    #self.fc1 = nn.Linear(3*3*128,128)
    #self.fc2 = nn.Linear(100,512)

  def forward(self, x, y):

    x = self.net1(x)
    y = self.net2(y)
    m = torch.cat((x, y), 1)
    m = self.dropout(m)
    m = self.fc(m)
    return m


class Net(nn.Module):
  def __init__(self, hsi_channels, sar_channels, dsm_channels, patch_size, num_class):
    super(Net, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.net1 = lidar_branch(sar_channels)
    self.net2 = hsi_branch(hsi_channels, patch_size)
    self.net3 = lidar_branch(dsm_channels)

    self.dropout = nn.Dropout(p=0.5)
    self.fc = nn.Linear(384, num_class)

    #self.fc1 = nn.Linear(3*3*128,128)
    #self.fc2 = nn.Linear(100,512)

  def forward(self, x, y, l, dsm):

    l = self.net1(l)
    y = self.net2(x, y)
    dsm = self.net3(dsm)
    m = torch.cat((l, y, dsm), 1)
    m = self.dropout(m)
    m = self.fc(m)

    return m