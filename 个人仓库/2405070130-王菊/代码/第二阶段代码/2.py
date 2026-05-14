'''
#激活函数实践
import torch
import torch.nn as nn
x=torch.tensor([-1.0,-0.5,0.0,0.5,1.0])
print(f"原始输入{x}")

#Sigmoid输出0-1
sigmoid = nn.Sigmoid()
out_sigmoid = sigmoid(x)
print("Sigmoid输出",out_sigmoid)

#ReLU,正的留下，负的归零
relu=nn.ReLU()
out_relu=relu(x)
print("ReLU输出",out_relu)

#GELU 负数不会直接变 0，而是慢慢变小
gelu=nn.GELU()
out_gelu=gelu(x)
print("GELU输出",out_gelu)

#SwiGLU 把输入切成两半，一半做门控，效果最好
class SwiGLU(nn.Module):
    # 前向传播函数：数据进入网络后执行的计算逻辑
    def forward(self, x):
        # 沿着最后一个维度 dim=-1，将输入 x 切分成 2 等份
        # x1：前一半数据，直接参与后续计算
        # x2：后一半数据，经过 sigmoid 后作为门控信号
        x1, x2 = x.chunk(2, dim=-1)
        # SwiGLU 核心公式：前半部分 × 门控值（sigmoid(后半部分)）
        # torch.sigmoid(x2)：将 x2 压缩到 0~1 之间，起到控制信息流通的门控作用
        return x1 * torch.sigmoid(x2)

swiglu = SwiGLU()
# SwiGLU 输入必须是偶数维度，我们重新造数据
x_swiglu = torch.tensor([-1.0, -0.5, 0.0, 0.5])
out_swiglu = swiglu(x_swiglu)
print("SwiGLU 输出：", out_swiglu)
'''
#在神经网络上实战
import torch
import torch.nn as  nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10,5)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.linear(x)
        x=self.relu(x)
        return x

x=torch.randn(1,10)
net=Net()
out=net(x)
print(out)




