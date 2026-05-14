# 点积
import torch
y = torch.ones(4, dtype= torch.float32)
x = torch.arange(4,dtype= torch.float32)
x_y_1 = torch.dot(x,y)
x_y_2 = torch.sum(x*y)
print(x_y_1, x_y_2)


A = torch.ones(16, dtype=torch.float32).reshape((4,4))
B = torch.arange(16.0).reshape((4,4))
u = torch.arange(4.0)
Ax_mv = torch.mv(A, x)
AB_mm = torch.mm(A, B)
u_2 = torch.norm(u)
u_1 = torch.abs(u).sum()
fln = torch.norm(torch.ones((4,9)))


print(Ax_mv)
print(Ax_mv.shape,Ax_mv.dtype)


print(AB_mm)
print(AB_mm.shape, AB_mm.dtype)

print(u_2.item())
print(u_1.item())
print(fln.item())
