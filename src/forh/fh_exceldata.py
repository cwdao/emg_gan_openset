# -*- coding:utf-8 -*-
# 加载一些包
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# 设置device为cuda，也就是gpu，如果没有gpu则使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 自定义神经网络,CNN,当然现在没有卷积层，也可以叫NN
# 输入数据的尺寸
hdreshape = 8
# 隐藏层1神经元个数
hdlayer_1 = 16
# 隐藏层2神经元个数
hdlayer_2 = 8
# 隐藏层3神经元个数
hdlayer_3 = 256


# 建立pytorch的神经网络类，可以看到基于nn.moudle生成的，包含初始化_init_,前向传播（也就是网络结构）
# forward 中有很多注释掉的层，实际上我们正是在这里修改网络结构，目前我只用到了fc1,fc2和out三个。所有的网络层都需要先在初始化定义好
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=0
        )

        self.fc1 = nn.Linear(in_features=hdreshape, out_features=hdlayer_1)
        self.fc2 = nn.Linear(in_features=hdlayer_1, out_features=hdlayer_2)
        # self.fc3 = nn.Linear(in_features=hdlayer_2, out_features=hdlayer_3)
        self.out = nn.Linear(in_features=hdlayer_2, out_features=1)
        self.dr1 = nn.Dropout2d(0.2)

    def forward(self, t):
        # (1) input layer
        t = t
        # t = t.reshape(5,12)
        # t = t.unsqueeze(0)

        # (2) hidden conv layer
        # t = self.conv1(t)
        # t = F.relu(t)
        # t = F.max_pool2d(t, kernel_size=2, stride=1)

        # (3) hidden conv layer
        # t = self.conv2(t)
        # t = F.relu(t)
        # t = self.dr1(t)
        # t = F.max_pool2d(t, kernel_size=2, stride=1)

        # (4) hidden linear layer
        # t = t.reshape(-1, hdreshape)
        # t = t.flatten(start_dim=0)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        # t = self.fc3(t)
        # t = F.relu(t)
        # t = self.dr1(t)

        # (5) output layer
        t = self.out(t)

        return t


# 基于Network 类生成net对象
net = Network()

# 打印网络，检查输入输出 shape是否正确
# print(net)
summary(net, (1, 1, 8), batch_size=1, device="cpu")
# 可视化结构，torchviz
sampleInput = torch.randn(1, 1, 1, 8).requires_grad_(True)
sampleOutput = net(sampleInput)


# 读取xlsx文件，这部分直接使用了之前的代码./src/forh/dataset.xlsx
data = pd.read_excel("C:/Users/cwdbo/OneDrive/桌面/theTEMP/技术/torchlearn/src/forh/dataset.xlsx")
# C:\Users\cwdbo\OneDrive\桌面\theTEMP\技术\torchlearn\src\forh\dataset.xlsx
# 转为 numpy array格式，方便后续数据处理
data = np.array(data)
# 转为 tensor 格式，以允许 pytorch 使用
data = torch.tensor(data)

# 划分训练集和测试集
data_train = data[0:7, :]
data_train_X = data_train[:, 0:8]
data_train_Y = data_train[:, 8]

data_test = data[7:, :]
data_test_X = data_test[:, 0:8]
data_test_Y = data_test[:, 8]


# 使用TensorDataset函数构建pytorch数据集
dataset_train = torch.utils.data.TensorDataset(data_train_X, data_train_Y)
dataset_test = torch.utils.data.TensorDataset(data_test_X, data_test_Y)
train_set = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
test_set = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)
# 测试一下数据集，这里让train_set输出一个样本（包含样本X和标签Y），看看构建的是否正确
sample = next(iter(train_set))
print(sample)


# CNN 网络加载
net = Network()

# 损失函数设置为MSE，也就是均方根误差
criterion = torch.nn.MSELoss()


# 加载数据，设置优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# lr_schedule = torch.optim.lr_scheduler.StepLR(\
#         optimizer, 1, gamma=0.8, last_epoch=-1)

# 训练过程
# 设置总训练轮次
epoch_num = 2000
# 把网络送到device(如果有GPU)
net.to(device)
# 把网络设置为训练模式，这是因为一些特殊层（例如dropout）在训练和测试使用的时候需要不同的特性
net.train()
# 开始训练
for epoch in range(epoch_num):

    # 从训练集读取一个batch，batch大小由自己设置
    for batch in train_set:
        datas, labels = batch
        # 需要转为float tensor才能训练
        datas = datas.to(torch.float32)
        labels = labels.float()
        # 样本datas输入net,成为preds。加上.to(device)是为了确保它在GPU运行
        preds = net(datas.to(device))
        # 训练损失的计算由刚刚定义的损失函数负责，需要网络预测的结果preds，和真实数据labels，度量它们的距离
        trainloss = criterion(preds.to(device), labels.to(device))
        # 固定步骤，更新网络参数前先清零，避免叠加；然后反向传播，再更新参数
        optimizer.zero_grad()
        trainloss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weight
        # lr_schedule.step() # 学习率变化

    # 每50个epoch打印一次当前的损失，方便观察
    if (epoch + 1) % 50 == 0:
        print(
            "epoch",
            epoch + 1,
            "MSE_tr:",
            float(trainloss),
        )


# 验证效果
# 加载测试集样本，然后与预测的结果比较。
for batch in test_set:

    test_data_sample_X, test_data_sample_Y = batch
    # 把测试X输入网络
    net.eval()
    net.to("cpu")
    # test_data_sample_X = test_data_sample_X.float()
    predict = net(test_data_sample_X.float())

    print(test_data_sample_Y, predict)
