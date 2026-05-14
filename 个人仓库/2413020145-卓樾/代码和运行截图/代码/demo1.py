import torch

matrix = torch.tensor([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12],
    [13,14, 15, 16]
])

print(matrix)
print("="*40)

#一个元素
element = matrix[1, 2]
print(element.item())

#一行
row = matrix[1, :]
print(row.tolist())

#一列
col = matrix[:, 1]
print(col.tolist())

#子区域
sub1 = matrix[1:3, 1:]
print(sub1)

#子区域
sub2 = matrix[::3, ::2]
print(sub2)

