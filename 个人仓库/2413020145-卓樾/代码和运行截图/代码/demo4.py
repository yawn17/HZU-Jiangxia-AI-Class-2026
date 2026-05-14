import torch
#将大小为1的张量转换为Python标量
a = torch.tensor([3.5])
b = (a, a.item(), float(a), int(a))
print(b)