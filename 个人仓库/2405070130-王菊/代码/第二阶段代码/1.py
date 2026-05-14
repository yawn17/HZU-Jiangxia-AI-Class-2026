
import torch
import torch.nn as nn
#linear的实践
#创建线性算子，输入10个特征-》输出5个特征
linear = nn.Linear(in_features=10,out_features=5)
#构造输入数据，1条数据，每条10个特征
x1=torch.randn(1,10)
#使用算子计算
output1 = linear(x1)
print(f"输入形状{x1.shape},输出形状{output1.shape}")

#conv卷积算子的实践
#输入1通道——》输出32通道，卷积核3*3
conv=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3)
#输入一张图片，一个通道，高度28*28
x2=torch.randn(1,1,28,28)
#卷积计算
output2=conv(x2)
print(f"输入图像形状{x2.shape},输出形状{output2.shape}")

#mn矩阵乘法算子
A=torch.randn(2,3)
B=torch.randn(3,4)
#矩阵乘法
C=torch.matmul(A,B)
print(f"A的形状{A.shape}，B的形状{B.shape}，C的形状{C.shape}")

#LSTM时序算子
#输入特征10-》隐藏层20
lstm=nn.LSTM(input_size=10,hidden_size=20)
#输入：序列长度5，批量1，每个数据10个特征
x3=torch.randn(5,1,10)
#运行LSTM
output3,(h_n,c_n)=lstm(x3)
print(f"输入形状{x3.shape},LSTM输出形状{output3.shape}")

