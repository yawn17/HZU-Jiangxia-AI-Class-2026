import torch

#矩阵的转置
A = torch.arange(20).reshape(5,4)
print(A, A.T, sep="\n")

#访问长度
print(A[3], len(A), A.shape, sep="\n")

#指定轴的求和汇总张量
A = torch.arange(20*2).reshape(2,5,4)
A_sum_aixs0 = A.sum(dim=0)
A_sum_aixs1 = A.sum(dim=1)
A_sum_aixs2 = A.sum(dim=2)
A_sum_aixs01 = A.sum(dim=[0,1])
sum_A = A.sum(axis=2, keepdims=True)
print(A, A.shape)
print(A_sum_aixs0, A_sum_aixs0.shape)
print(A_sum_aixs1, A_sum_aixs1.shape)
print(A_sum_aixs2, A_sum_aixs2.shape)
print(A_sum_aixs01, A_sum_aixs01.shape)
print(sum_A, sum_A.shape)

#指定轴的累积总和
A_cumsum_axis0 = A.cumsum(axis=0)
print(A_cumsum_axis0, A_cumsum_axis0.shape)