import torch

# 张量创建
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("原始张量：")
print(x)

# 索引与切片
print("\n第一行：", x[0])
print("第一列：", x[:, 0])

# 广播
y = torch.tensor([1,2,3,4])
print("\n广播相加：")
print(x + y)

# 简单运算
a = torch.tensor([3.0])
b = a ** 2
print("\na = 3, a² =", b.item())