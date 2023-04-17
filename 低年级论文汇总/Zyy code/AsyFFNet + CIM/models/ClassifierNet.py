import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.fc2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc2(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        out = out * x + x
        return out

class CIM(nn.Module):
    def __init__(self, hidden_size):
        super(CIM, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, 128, 1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, 128, 1)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.ca = ChannelAttention(128)

    def forward(self, x, y, z):
        s1 = x + y
        s1 = self.conv1(s1)
        s1 = self.bn1(s1)
        s1 = F.relu(s1)
        s1 = self.ca(s1)

        s2 = self.conv2(z)
        s2 = self.bn2(s2)
        s2 = F.relu(s2)
        return s1 + s2


class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_in3, size_out=16):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_out = size_in1, size_in2, size_in3, size_out

        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        #self.hidden_sigmoid = nn.Linear(size_out * 3, 3, bias=False)
        self.tanh_f = nn.Tanh()
        #self.sigmoid_f = nn.Sigmoid()
        self.softmax_f = nn.Softmax(dim=2)

    def forward(self, x1, x2, x3):
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x1.size(0), -1)
        x3 = x3.reshape(x1.size(0), -1)
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        h3 = self.tanh_f(self.hidden3(x3))
        #x = torch.cat((h1, h2, h3), dim=1)
        x = torch.stack((h1, h2, h3), dim=2).unsqueeze(3)
        #z = self.softmax_f(self.hidden_sigmoid(x))
        z = self.softmax_f(x)
        #z = z.view(z.size(0), -1)
        '''
        for i in range(z.size(0)):
            sum1, sum2, sum3 = 0, 0, 0
            for j in range(self.size_out):
                sum1 += z[i, j, 0, 0]
                sum2 += z[i, j, 1, 0]
                sum3 += z[i, j, 2, 0]
            print('sum1 =', sum1)
            print('sum2 =', sum2)
            print('sum3 =', sum3)
        '''
        out = z[:, :, 0].squeeze(2) * h1 + z[:, :, 1].squeeze(2) * h2 + z[:, :, 2].squeeze(2) * h3
        #print(out.shape)
        return out

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class Net(nn.Module):
    def __init__(self, hsi_channels, sar_channels, dsm_channels, hidden_size, patch_size, num_classes=7):
        self.planes = hidden_size
        self.num_parallel = 3

        super(Net, self).__init__()

        self.conv_x = nn.Conv2d(hsi_channels, hidden_size, 3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(sar_channels, hidden_size, 3, padding=1, bias=False)
        self.conv_z = nn.Conv2d(dsm_channels, hidden_size, 3, padding=1, bias=False)
        self.bnx = nn.BatchNorm2d(hidden_size)
        self.bny = nn.BatchNorm2d(hidden_size)
        self.bnz = nn.BatchNorm2d(hidden_size)

        self.cim = nn.ModuleList([CIM(self.planes) for _ in range(15)])

        self.conv_out = nn.ModuleList([nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False) for _ in range(4)])
        self.conv_sar1 = nn.ModuleList([nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False) for _ in range(4)])
        self.conv_sar2 = nn.ModuleList([nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False) for _ in range(4)])
        self.bn_out = nn.ModuleList([nn.BatchNorm2d(hidden_size) for _ in range(4)])
        self.bn_sar1 = nn.ModuleList([nn.BatchNorm2d(hidden_size) for _ in range(4)])
        self.bn_sar2 = nn.ModuleList([nn.BatchNorm2d(hidden_size) for _ in range(4)])

        self.gml = GatedMultimodalLayer(hidden_size * patch_size * patch_size, hidden_size * patch_size * patch_size, hidden_size * patch_size * patch_size)

        self.classifier = Classifier(num_classes)

    def forward(self, x, y, z):
        x = self.conv_x(x)
        x = self.bnx(x)
        x = F.relu(x)
        y = self.conv_y(y)
        y = self.bny(y)
        y = F.relu(y)
        z = self.conv_z(z)
        z = self.bnz(z)
        z = F.relu(z)
        out1 = self.cim[0](x, y, z)
        out2 = self.cim[1](x, z, y)
        out3 = self.cim[2](y, z, x)
        for i in range(4):
            out1 = self.conv_out[i](out1)
            out1 = self.bn_out[i](out1)
            out1 = F.relu(out1)
            out2 = self.conv_sar1[i](out2)
            out2 = self.bn_sar1[i](out2)
            out2 = F.relu(out2)
            out3 = self.conv_sar2[i](out3)
            out3 = self.bn_sar2[i](out3)
            out3 = F.relu(out3)
            x = out1
            y = out2
            z = out3
            out1 = self.cim[3 * (i + 1)](x, y, z)
            out2 = self.cim[3 * (i + 1) + 1](x, z, y)
            out3 = self.cim[3 * (i + 1) + 2](y, z, x)
        out = self.gml(out1, out2, out3)
        out = self.classifier(out)
        return out, out1, out2, out3
