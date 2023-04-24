import os
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch, math
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat, savemat
import random
from time import time
import h5py


"""## **1.定义网络结构**

使用二维卷积操作提取图像的空间特征
"""


#########################################

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
  def __init__(self):
    super(cascadeNet, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.conv0 = nn.Conv2d(8, 64, 3, padding=1)  # YC
    self.cascade_block1 = cascade_block(64)

    self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    self.cascade_block2 = cascade_block(128)

    self.conv2 = nn.Conv2d(256, 128, 3)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 128, 3)
    self.bn3 = nn.BatchNorm2d(128)
    self.pool = nn.AvgPool2d(5)

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

    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = F.relu(self.conv2(x))
    x = self.max_pool(x)
    x1 = x.contiguous().view(x.size(0), -1)

    return x1


class simple_cnn_branch(nn.Module):
  def __init__(self):
    super(simple_cnn_branch, self).__init__()

    self.conv1 = nn.Conv2d(166, 256, 3, padding=1)  # YC
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
  def __init__(self):
    super(lidar_branch, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.net = cascadeNet()

    self.dropout = nn.Dropout(p=0.5)
    self.fc = nn.Linear(128, 128)

  def forward(self, x):

    x = self.net(x)
    x = self.dropout(x)
    x = self.fc(x)

    return x


class hsi_branch(nn.Module):
  def __init__(self):
    super(hsi_branch, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.net1 = simple_cnn_branch()
    self.net2 = pixel_branch()

    self.dropout = nn.Dropout(p=0.5)
    self.fc = nn.Linear(5664, 128)

    #self.fc1 = nn.Linear(3*3*128,128)
    #self.fc2 = nn.Linear(100,512)

  def forward(self, x, y):

    x = self.net1(x)
    y = self.net2(y)
    m = torch.cat((x, y), 1)
    m = self.dropout(m)
    m = self.fc(m)
    return m


class finetue_Net(nn.Module):
  def __init__(self):
    super(finetue_Net, self).__init__()

   #self.conv1 = nn.Conv2d(285, 256, 3, padding=1)#hhk
    self.net1 = lidar_branch()
    self.net2 = hsi_branch()

    self.dropout = nn.Dropout(p=0.5)
    self.fc = nn.Linear(256, 5)

    #self.fc1 = nn.Linear(3*3*128,128)
    #self.fc2 = nn.Linear(100,512)

  def forward(self, l, x, y):

    l = self.net1(l)
    y = self.net2(x, y)
    m = torch.cat((l, y), 1)
    m = self.dropout(m)
    m = self.fc(m)

    return m

train_hsiCube = np.load("D:\Compare/twobranch_torch//train_hsiCube_11.npy")
train_patches = np.load("D:\Compare/twobranch_torch//train_patches_11.npy")
train_labels = np.load("D:\Compare/twobranch_torch//train_labels.npy")
train_hsiCube_1 = np.load("D:\Compare/twobranch_torch//train_hsiCube_1.npy")

val_hsiCube = np.load("D:\Compare/twobranch_torch//val_hsiCube_11.npy")
val_patches = np.load("D:\Compare/twobranch_torch//val_patches_11.npy")
val_labels = np.load("D:\Compare/twobranch_torch//val_labels.npy")
val_hsiCube_1 = np.load("D:\Compare/twobranch_torch//val_hsiCube_1.npy")

train_hsiCube = torch.from_numpy(train_hsiCube.transpose(0, 3, 1, 2)).float()
train_hsiCube_1 = torch.from_numpy(train_hsiCube_1.transpose(0, 3, 1, 2)).float()
train_patches = torch.from_numpy(train_patches.transpose(0, 3, 1, 2)).float()

val_hsiCube = torch.from_numpy(val_hsiCube.transpose(0, 3, 1, 2)).float()
val_hsiCube_1 = torch.from_numpy(val_hsiCube_1.transpose(0, 3, 1, 2)).float()
val_patches = torch.from_numpy(val_patches.transpose(0, 3, 1, 2)).float()

print (train_hsiCube.shape)
print (val_hsiCube.shape)


class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = train_labels.shape[0]
        self.hsi = torch.FloatTensor(train_hsiCube)
        self.hsi_1 = torch.FloatTensor(train_hsiCube_1)
        self.lidar = torch.FloatTensor(train_patches)
        self.labels = torch.LongTensor(train_labels - 1)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.hsi[index], self.hsi_1[index], self.lidar[index], self.labels[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len


class ValDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = val_labels.shape[0]
        self.hsi = torch.FloatTensor(val_hsiCube)
        self.hsi_1 = torch.FloatTensor(val_hsiCube_1)
        self.lidar = torch.FloatTensor(val_patches)
        self.labels = torch.LongTensor(val_labels - 1)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.hsi[index], self.hsi_1[index], self.lidar[index], self.labels[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# 创建 trainloader 和 testloader
trainset = TrainDS()
valset = ValDS()


train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 64, shuffle = True, num_workers = 0)
val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=64, shuffle=True, num_workers=0)

"""# **4.定义模型的损失函数、训练和测试函数**"""


def calc_label_sim(label_1, label_2):
    sim = label_1.unsqueeze(1).float().mm(label_2.unsqueeze(1).float().t())
    return sim


criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  total_loss = 0
  for i, (inputs_1, inputs_2, inputs_3, labels) in enumerate(train_loader):
        inputs_1, inputs_2, inputs_3 = inputs_1.to(device), inputs_2.to(device), inputs_3.to(device)

        labels = labels.to(device)
        b, c, h, w = inputs_2.size()

        inputs_2 = inputs_2.view(b, c, -1)
        inputs_2 = inputs_2.permute(0, 2, 1)

        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化 
        outputs = model(inputs_3, inputs_1, inputs_2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

  print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))


def val(model, device, val_loader):
  model.eval()
  count = 0
  feature = []
  flabel = []
# 模型测试
  for inputs_1, inputs_2, inputs_3, labels in val_loader:

    inputs_1, inputs_2, inputs_3 = inputs_1.to(
        device), inputs_2.to(device), inputs_3.to(device)
    b, c, h, w = inputs_2.size()

    inputs_2 = inputs_2.view(b, c, -1)
    inputs_2 = inputs_2.permute(0, 2, 1)
    outputs = model(inputs_3, inputs_1, inputs_2)
    #feature.append(outputs.detach().cpu().numpy())
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

    if count == 0:
        y_pred_test = outputs
        test_labels = labels
        count = 1
    else:
        y_pred_test = np.concatenate((y_pred_test, outputs))
        test_labels = np.concatenate((test_labels, labels))
  #sio.savemat('feature.mat', {'feature': feature})
  a = 0
  for c in range(len(y_pred_test)):
    if test_labels[c] == y_pred_test[c]:
      a = a+1
  score = a/len(y_pred_test)*100
  print('%.2f' % (score))
  return score

"""# **5.开始训练和测试**"""

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
model = finetue_Net().to(device)

lr = 0.001
momentum = 0.9
# Adam参数设置
betas = (0.9, 0.999)

params_to_update = list(model.parameters())

# optimizer = torch.optim.Adam(params_to_update, lr=lr, betas=betas, eps=1e-8, weight_decay=0.0005)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0005)

num_epochs = 150
best_acc = 0
for epoch in range(num_epochs):
  train(model, device, train_loader, optimizer, epoch)
  if epoch % 1 == 0:
      start = time()
      score = val(model, device, val_loader)
      end = time()
      # print(end-start)
      # print(score)
      if score >= best_acc:
          best_acc = score
          print("save model")
          torch.save(model.state_dict(),'./model/model_'+str(epoch)+'.pth')


score = val(model, device, val_loader)
print(score)


print("chenggong!")
