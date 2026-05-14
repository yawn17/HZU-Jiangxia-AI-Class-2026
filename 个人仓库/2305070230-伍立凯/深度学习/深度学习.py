    #自动求导
# import torch
# def get_hanshu():   #获取函数
#     hanshu=input("输入函数:(x**3+4*x**2+8*x+5)")
#     def get_eval(x):
#         return eval(hanshu,{"x":x})
#     return get_eval

# def get_x():   #获取求导的x值
#     x=float(input("输入x:(3)"))
#     return x


# def hanshuqiudao(hanshu,x):   #对函数求导
#     x1=torch.tensor(x,dtype=torch.float32, requires_grad=True)
#     y=hanshu(x1)
#     y.backward()
#     return x1.grad.item()

# hanshu=get_hanshu()
# x=get_x()

# print(hanshuqiudao(hanshu,x))







# 张量克隆（分配新内存）
# import torch
# old_tensor = torch.tensor([1, 2, 3])
# new_tensor = old_tensor.clone()
# new_tensor[0] = 0
# print("老张量:", old_tensor)
# print("新张量:", new_tensor)



# 创建张量
# import torch
# x = torch.tensor([1, 2, 3])
# print(x)



# 从0开始顺序创建张量
# import torch
# x = torch.arange(5)
# print(x)



# 按维度累积求和
# import torch
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print("原始张量:\n", x)
# print("按行累积求和 (dim=0):\n", x.cumsum(dim=0))
# print("按列累积求和 (dim=1):\n", x.cumsum(dim=1))



# 向量点积（按元素相乘再求和）
# import torch
# x = torch.tensor([1, 2, 3])
# y = torch.tensor([4, 5, 6])
# print("torch.dot(x, y):", torch.dot(x, y))



# 矩阵乘向量 (matrix-vector product)
# import torch
# A = torch.tensor([[1, 2], [3, 4]])
# v = torch.tensor([5, 6])
# print(torch.mv(A, v))



# 矩阵乘矩阵 (matrix-matrix product)
# import torch
# A = torch.tensor([[1, 2], [3, 4]])
# B = torch.tensor([[5, 6], [7, 8]])
# print(torch.mm(A, B))



# 计算张量的L2范数
# import torch
# x = torch.tensor([3.0, 4.0])
# print("L2范数:", torch.norm(x))



# 计算张量的L1范数
# import torch
# x = torch.tensor([-1, 2, -3])
# print("L1范数:", torch.abs(x).sum())



# 张量转置
# import torch
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print("原始张量:\n", x)
# print("转置后:\n", x.T)



# 获取张量的形状
# import torch
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print("张量形状:", x.shape)



# 张量的元素总数
# import torch
# x = torch.arange(12)
# print(x.numel())



# 改变张量形状（不改变元素数量和值）
# import torch
# old_tensor = torch.arange(6)
# new_tensor = old_tensor.reshape(2, 3)
# print("老张量:", old_tensor)
# print("新张量:\n", new_tensor)



# 张量拼接
# import torch
# x = torch.tensor([[1, 2], [3, 4]])
# y = torch.tensor([[5, 6], [7, 8]])
# print("按行堆叠 (dim=0):\n", torch.cat((x, y), dim=0))
# print("按列堆叠 (dim=1):\n", torch.cat((x, y), dim=1))



# 逐元素比较生成布尔张量
# import torch
# x = torch.tensor([1, 2, 3])
# y = torch.tensor([1, 2, 4])
# print(x == y)



# 按维度求和（keepdims控制是否保持维度）
# import torch
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print("按行求和 (dim=0, keepdims=False):", x.sum(dim=0))
# print("按列求和 (dim=1, keepdims=True):\n", x.sum(dim=1, keepdims=True))



# 按维度求均值（keepdims控制是否保持维度）
# import torch
# x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print("按行求均值 (dim=0):", x.mean(dim=0))
# print("按列求均值 (dim=1, keepdims=True):\n", x.mean(dim=1, keepdims=True))



# 广播机制示例
# import torch
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# y = torch.tensor([10, 20, 30])
# print(x + y)



# 多元素索引赋值
# import torch
# x = torch.zeros(5)
# x[[0, 2, 4]] = 1
# print(x)



# 张量与其他格式转换
# import torch
# x = torch.tensor([1, 2, 3])
# numpy_arr = x.numpy()
# print("numpy数组:", numpy_arr)
# y = torch.tensor([5])
# print("标量张量转int:", int(y))



# 反向传播求梯度
# import torch
# x = torch.tensor(2.0, requires_grad=True)
# y = x ** 2
# y.backward()
# print("x的梯度:", x.grad)