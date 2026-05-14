# %% [markdown]
# conda env remove d2l-zh
# conda create -n -y d2l-zh python=3.8 pip
# conda activate d2l-zh
# 
# pip install -y jupyter d2l torch torchvision
# 
# wget https://zh-v2.d2l.ai/d2l-zh.zip
# unzip d2l-zh.zip
# jupyter notebook

# %%
##########  访问数组里的元素
#   一个，行列，集中、分散子区域
import numpy as np
a = np.array([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16]
])
print(a[-1])
print(a[1,2],a[1,:],a[:,1],a[1:3,1:],a[::3,::2],sep="\n")



# %%
##########  创建张量
#   arange(s), zeros((x,y,h)), ones((x,y,h)), tensor([ [2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1] ]), 逻辑运算符, randa(,), sum()
print("创建张量")
import torch
x1= torch.arange(12)
x1_1  = torch.arange(12.0) # 加.0自动为float类型
x2= torch.zeros((2,3,4))
x3= torch.ones((2,3,4))
x4= torch.tensor([ [2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1] ])
x5= torch.randn(4, 4)   # 生成随机的 标准正态分布（均值 μ=0，标准差 σ=1）【数值大部分集中在 -2 ~ 2 之间（约 95%）】
x5_1= torch.randn(4, 4, dtype=torch.float32)  # 显式指定float32
x5_2= torch.randn(4, 4).float()
X= x2==x3

print(x1.sum(),x2.sum(), x3.sum())
print(x1, x1_1, x2, x3, x4, X, x5, x5_1, sep="\n")


##########  张量运算【加减乘除】
print("张量运算")
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

x1 = x + y
x2 = x - y
x3 = x * y
x4 = x / y
print(x, y, x1, x2, x3, x4, sep="\n")

##########  张量连结
print("张量连结")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
x5 = torch.cat((X, Y),dim=0)
x6 = torch.cat((X, Y),dim=1)
print(X, Y, x5, x6, sep="\n")



# %%
##########  张量广播机制
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
c = a+b
print(a, b, c, sep="\n")

# %%
##########  访问张量的形状和元素总数,改变张量形状
#   shape,numel()，reshape(x,y,...)、reshape((1,-1))、reshape((-1,1))
x1 = torch.arange(12)
print(x1,x1.shape,x1.numel(),sep="\n")
x2 = x1.reshape(3,4)
x3 = x1.reshape((-1,1))
x4 = x1.reshape((1,-1))
x2, x3, x4

# %%
##########  写入张量
print("写入张量")
X = torch.arange(12).reshape((3,4))
print(X)
print(X[1,2])
X[1,2] = 9
print(X[1,2])

print(X)
print(X[1:2,:])
X[1:2,:] = 12
print(X[1:2,:])

##########  转化为numpy张量，大小为 1 的张量转 Python 标量
print("转化为numpy张量")
A = X.numpy()
# 将 NumPy 数组 A 转换回 PyTorch 张量
B = torch.tensor(A)
# 查看两者的数据类型
print(type(A), type(B), sep="\n")

print("大小为 1 的张量转 Python 标量")
# 创建一个仅包含单个元素的张量
a = torch.tensor([3.5])
# 多种方式提取为 Python 标量
print(a, a.item(), float(a), int(a), sep="\n")


# %% [markdown]
# ### 第一段：验证运算后张量地址变化
# ```python
# # 记录 Y 原来的内存地址
# before = id(Y)
# # 执行加法运算并重新赋值
# Y = Y + X
# # 比较运算前后 Y 的内存地址是否相同
# id(Y) == before
# ```
# 
# ---
# 
# ### 第二段：执行原地操作（保持内存地址不变）
# ```python
# # 创建一个和 Y 形状、数据类型相同的全零张量 z
# z = torch.zeros_like(Y)
# # 打印 z 初始的内存地址
# print('id(z):', id(z))
# 
# # 使用切片语法, 执行原地赋值，不会创建新张量
# z[:] = X + Y
# # 再次打印 z 的内存地址，验证地址未变
# print('id(z):', id(z))
# 
# 
# #  X += Y 也是原地操作
# before = id(X)
# X += Y
# id(X) == before
# ```
# 

# %%
########### 数据预处理
#   建立文件夹（ os.makedirs(,) ）、文件（ os.path.join(,,) ）
import os
os.makedirs(os.path.join("..","data"),exist_ok=True)
data_file = os.path.join("..","data","house_tiny.cvs")
with open(data_file,"w") as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

#   数据加载、读取（ read_csv() ）
import pandas as pd
data = pd.read_csv(data_file)
print(data)

# %%
##########  处理数据-【缺失数据】
#   插值法（ .fillna( . ) ）
inputs, outputs = data.iloc[:, 0:2], data.iloc[:,2]
inputs = inputs.fillna( inputs.mean(numeric_only=True) )
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
inputs = inputs.astype(int)
print(inputs)

# %%
##########   inputs和outputs中所有条目都是数值类型，就可转化为张量格式
#   数值型dataframe转化为张量
import torch
x,y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x,y,sep="\n")

# 将大小为1的张量转换为Python标量
a = torch.tensor([3.5]) # 创建一个大小为1的张量
result = (a, a.item(), float(a), int(a)) # 多种方式将其转为Python标量
print(result)

# %%
##########  线性代数实现

#   通过指定两个分量m和n来创建一个形状为m*n的矩阵, 转置
import torch
x = torch.arange(20).reshape(5,4)
# print(x, x.T, sep="\n")

#   访问张量的长度
# print( x[3], len(x), x.shape, sep="\n")

#   指定轴的求和汇总张量【求平均值同理，只需sum改成mean】
A = torch.arange(20*2).reshape(2,5,4)
A_sum_aixs0 = A.sum(dim=0)  #  各批融合
A_sum_aixs1 = A.sum(dim=1)  #   各行融合
A_sum_aixs2 = A.sum(dim=2)  #   各列融合
A_sum_aixs01 = A.sum(dim=[0,1]) #   先批融合，再行融合
sum_A = A.sum(axis=2, keepdims=True) #计算总和或均值时保持轴数不变,要融合的轴变为1

print(A, A.shape)
# print(A_sum_aixs0, A_sum_aixs0.shape)
# print(A_sum_aixs1, A_sum_aixs1.shape)
# print(A_sum_aixs2, A_sum_aixs2.shape)
# print(A_sum_aixs01, A_sum_aixs01.shape)
# print(sum_A, sum_A.shape)

#   指定轴的累积总和
A_cumsum_axis0 = A.cumsum(axis=0)
print(A_cumsum_axis0, A_cumsum_axis0.shape)


# %%
#   点积:   各相同位置的元素相乘之和
import torch
y = torch.ones(4, dtype= torch.float32)
x = torch.arange(4,dtype= torch.float32)
x_y_1 = torch.dot(x,y) # 点积 法1 dot(,)
x_y_2 = torch.sum(x*y) # 点积 法2 先元素乘法，再进行求和 
print(x_y_1, x_y_2)

#   Ax, AB, L2范数， L1范数, 弗罗尼乌斯范数
A = torch.ones(16, dtype=torch.float32).reshape((4,4))
B = torch.arange(16.0).reshape((4,4))
u = torch.arange(4.0)
Ax_mv = torch.mv(A, x) # 两个输入必须 dtype 一致
AB_mm = torch.mm(A, B) # 两个输入必须 dtype 一致
u_2 = torch.norm(u)    # 输入必须dtype为浮点 / 复数型
u_1 = torch.abs(u).sum()
fln = torch.norm(torch.ones((4,9)))


# %% [markdown]
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# B = A.clone() # 通过分配新内存，将A的一个副本分配给B
# A, A + B

# %%
########### 自动求导
#   对非标量调用“backward”需要传入一个“gradient”参数
import torch
x = torch.arange(4.0)

# 定义张量并开启梯度追踪
x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于 `x = torch.arange(4.0, requires_grad=True)`
print("x.grad (初始值):", x.grad)  # 初始为 None

# 计算 y
y = 2 * torch.dot(x, x)
print("y:", y)

# 定义张量并开启梯度追踪
x = torch.arange(4.0)
x.requires_grad_(True)
# 计算 y
y = 2 * torch.dot(x, x)

# 反向传播计算梯度
y.backward()
# 查看梯度
print("x.grad:", x.grad)
# 验证梯度是否等于 4 * x
print("x.grad == 4 * x:", x.grad == 4 * x)

# --- In [6] ---
# 第一次计算 y = 2 * x^2
y1 = 2 * torch.dot(x, x)
y1.backward()
print("第一次梯度:", x.grad)
# 清空梯度，计算新的函数 y = x.sum()
x.grad.zero_()  # 清除之前累积的梯度
y2 = x.sum()
y2.backward()
print("第二次梯度:", x.grad)

# --- In [7] ---
# 清空之前累积的梯度
x.grad.zero_()
# 计算 y = x * x（逐元素相乘）
y = x * x
# 对 y 求和后反向传播，等价于 y.backward(torch.ones(len(x)))
y.sum().backward()
# 查看梯度
print("x.grad:", x.grad)

# --- In [8] ---
# 清空梯度
x.grad.zero_()
# 计算 y = x * x
y = x * x
# 分离 y，得到 u，u 不再参与梯度计算
u = y.detach()
# 计算 z = u * x
z = u * x
# 对 z 求和后反向传播
z.sum().backward()
# 验证梯度是否等于 u
print("x.grad == u:", x.grad == u)  # 输出 tensor([True, True, True, True])

# --- In [9] ---
# 清空梯度
x.grad.zero_()
# 对 y 求和后反向传播
y.sum().backward()
# 验证梯度是否等于 2 * x
print("x.grad == 2 * x:", x.grad == 2 * x)  # 输出 tensor([True, True, True, True])

# %% [markdown]
# 即使构建函数的计算图需要通过 Python 控制流（如 while 循环和 if 条件或任意函数调用），我们仍然可以计算得到变量的梯度

# %%
import torch

# --- In [10] ---
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 创建标量张量 a，并开启梯度追踪
a = torch.randn(size=(), requires_grad=True)
# 计算 d = f(a)
d = f(a)
# 反向传播计算梯度
d.backward()

# 查看梯度
print("a.grad:", a.grad)
# 验证梯度是否等于 d / a
print("a.grad == d / a:", a.grad == d / a)


