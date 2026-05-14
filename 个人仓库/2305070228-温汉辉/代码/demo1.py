import torch;   
import numpy as np
x = torch.tensor([1,2,3,4,5])
x=torch.arange(12)
x=x.reshape(3,4)
x = torch.zeros(3,4)
x = torch.ones(3,4)
x=torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]]).reshape(2,6)
x=torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
y=torch.tensor([[1.1,2.2,3.3,4.4],[5.5,6.6,7.7,8.8],[9.9,10.10,11.11,12.12]])
x1=torch.arange(12,dtype=torch.float32).reshape(3, 4)
y1=torch.tensor([[1.1,2.2,3.3,4.4],[5.5,6.6,7.7,8.8],[9.9,10.10,11.11,12.12]])
result=torch.cat((x1,y1),dim=0)
result1=torch.cat((x1,y1),dim=1)
print(x1)
print(y1)
print(result)
print(result1)
print(x1==y1)
print(x1!=y1)
print(x1.sum())
x1[1,2]=9

print(x1)
x1[0:2,:]=12
print(x1)
A=x.numpy()
B=torch.tensor(A)
print(B)
print(type(A))
print(type(B))
a=torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))




