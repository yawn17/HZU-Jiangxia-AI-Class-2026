"""
    模型优化思路：
        1- 优化方法从SGD改为Adam
        2- 学习率
        3- 对数据进行标准化处理
        4- 增加网络深度，每层的神经元数量
        5- 调整训练的轮数
        ...
"""

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time, os

# 1 创建数据集
def create_dataset():
    # 1- 加载csv文件数据集
    data = pd.read_csv(r'.\data\手机价格预测.csv')
    # print(data)

    # 2- 获取x特征列和y标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # print(f"x: {x.head()}, {x.shape}")
    # print(f"y: {y.head()}, {y.shape}")

    # 3- 将特征列转化为浮点型
    x = x.astype(np.float32)

    # 4- 划分训练集和测试集
    # stratify=y: 划分数据集时，按照y标签列的分布进行划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

    # 5- 将数据转化为张量形式
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))

    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

# 2 创建模型
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 1- 定义隐藏层1,输入维度为input_dim,输出维度为128
        self.linear1 = nn.Linear(in_features=input_dim, out_features=128)

        # 2- 定义隐藏层2,输入维度为128,输出维度为256
        self.linear2 = nn.Linear(in_features=128, out_features=256)

        # 3- 定义输出层,输入维度为256,输出维度为output_dim
        self.output = nn.Linear(in_features=256, out_features=output_dim)


    def forward(self, x):
        # 1- 隐藏层1计算：加权求和 + 激活函数
        x = torch.relu(self.linear1(x))

        # 2- 隐藏层2计算：加权求和 + 激活函数
        x = torch.relu(self.linear2(x))

        # 3- 输出层计算：加权求和 + 激活函数
        # 正常写法，但不需要，后续使用多分类交叉熵损失函数CrossEntropyLoss时，会自动包含softmax计算
        # x = torch.softmax(self.output(x), dim=-1)
        x = self.output(x)

        return x

# 3 训练模型
def train():
    # 1- 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 2- 创建神经网络模型
    model = PhonePriceModel(input_dim, output_dim)

    # 3- 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 4- 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 5- 模型训练
    # 5.1 定义变量，记录训练的总轮次
    epochs = 50
    for epoch in range(epochs):
        # 5.2.1 定义变量，记录每轮训练的损失值，训练批次数
        total_loss, batch_num = 0.0, 0

        # 5.2.2 定义变量，表示本轮开始训练的时间
        start = time.time()
        # 5.2.3 开始本轮各批次的训练
        for x, y in train_loader:
            # 5.2.4 切换模型（状态）  train()
            model.train()
            # 5.2.5 模型预测
            y_pred = model(x)
            # 5.2.6 计算损失
            loss = criterion(y_pred, y)
            # 5.2.7 梯度清零，反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.2.8 累加损失值（把本轮每批次的平均损失累积起来）
            total_loss += loss.item()
            batch_num += 1

        # 5.2.4 本轮训练结束，打印训练信息
        print(f"epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time()-start:.2f}s")

    # print(f"模型的参数信息：{model.state_dict()}")
    # 6- 模型保存
    os.makedirs(r'.\model_data', exist_ok=True)
    torch.save(model.state_dict(), r'.\model_data\phone_price_model.pth')

# 4 模型测试
def evaluate():
    # 加载模型和训练好的参数
    model = PhonePriceModel(input_dim, output_dim)

    model.load_state_dict(torch.load(r'.\model_data\phone_price_model.pth'))

    # 创建测试集数据对象
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 定义变量，记录测试正确的样本数
    correct = 0

    # 从数据加载器中，获取到每批次的数据
    for x, y in test_loader:
        # 切换模型状态 -> 测试状态 eval()
        model.eval()

        # 模型预测
        y_pred = model(x)

        # 根据加权求和，得到类别，用argmax函数获取最大值的索引，即类别标签
        y_pred = torch.argmax(y_pred, dim=1)

        # 统计预测正确的样本数
        # print(y_pred == y)                # tensor([True, False, True,  ... , False, True, False])
        # print((y_pred==y).sum())          #
        correct += (y_pred == y).sum()

    print(f"准确率：{correct / len(test_dataset):.4f}")



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 1- 创建数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()

    # 2- 模型训练
    train()

    # 3- 模型测试
    evaluate()

