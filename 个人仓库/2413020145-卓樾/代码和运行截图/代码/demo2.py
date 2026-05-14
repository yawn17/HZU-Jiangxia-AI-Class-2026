import torch

#张量创建
x1 = torch.arange(12)
x1_1 = torch.arange(12.0)
x2 = torch.zeros((2, 3, 4))
x3 = torch.ones((2, 3, 4))
x4 = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
x5 = torch.randn(4, 4)
x5_1 = torch.randn(4, 4, dtype=torch.float32)
x5_2 = torch.randn(4, 4).float()
X_bool = x2 == x3

print(x1.sum(), x2.sum(), x3.sum(), sep=" | ")
print(x1)
print(x1_1)
print(x2)
print(x3)
print(x4)
print(X_bool)
print(x5)
print("\n")

#张量算术运算
x = torch.tensor([1.0, 2.0, 4.0, 8.0])
y = torch.tensor([2.0, 2.0, 2.0, 2.0])

add = x + y
sub = x - y
mul = x * y
div = x / y

print(x)
print(y)
print(add)
print(sub)
print(mul)
print(div)
print("\n")

#张量连结
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1.0, 4.0, 3.0],
                  [1.0, 2.0, 3.0, 4.0],
                  [4.0, 3.0, 2.0, 1.0]])

cat_row = torch.cat((X, Y), dim=0)
cat_col = torch.cat((X, Y), dim=1)

print(X)
print(Y)
print(cat_row)
print(cat_col)
print("\n")

#张量广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
c = a + b

print(a)
print(b)
print(c)
print("\n")

#张量形状操作
x1 = torch.arange(12)
print(x1)
print(x1.shape)
print(x1.numel())

x2 = x1.reshape(3, 4)
x3 = x1.reshape((-1, 1))
x4 = x1.reshape((1, -1))

print(x2)
print(x3)
print(x4)
print("\n")

#张量索引赋值
X = torch.arange(12).reshape((3, 4))
print(X)

print(X[1, 2])
X[1, 2] = 9
print(X[1, 2])

print(X[1:2, :])
X[1:2, :] = 12
print(X[1:2, :])
print(X)
print("\n")

#张量格式转换
A = X.numpy()
B = torch.tensor(A)

print(type(A))
print(type(B))
print(A)
print(float(A[0, 0]))
print(int(a[0].item()))
print(A.astype(float))
print(a.int())