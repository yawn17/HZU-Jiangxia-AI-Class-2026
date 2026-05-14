import torch
import os
import pandas as pd

'''1.
x = torch.arange(12)
print(x)
print("形状是:%s，元素总数是:%s" % (x.shape, x.numel()))
X1 = x.reshape(3,4)     #调用reshape函数
X2 = x.reshape(4,3)
print(X1,X2)
'''

'''2.
X1 =torch.zeros(2,1,2)   #全0
print(X1)
X2 =torch.ones(2,1,2)    #全1
print(X2)
X3 = torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
print(X3)
'''

'''3.
x = torch.tensor([[1,1],[2,2]])
y = torch.tensor([[2,2],[1,1]])
print(x+y,x-y,x*y,x/y,x**y)
print(torch.exp(x))
'''


'''4.#张量连结
X = torch.arange(12,dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(X.shape,Y.shape)
Z1 = torch.cat((X,Y),dim = 0)   #块数合并
Z2 = torch.cat((X,Y),dim = 1)   #行数合并
print(Z1.shape,Z2.shape)'''

'''
X1 = torch.ones(2,3,2)
X2 = torch.zeros(2,3,2)
Z3 = torch.cat((X1,X2),dim = 0)  #块数合并，增多块
print(X1.shape,X2.shape)
print(Z3.shape)
Z4 = torch.cat((X1,X2),dim = 1)  #行数合并，增多行
print(Z4.shape)
Z5 = torch.cat((X1,X2),dim = 2)   #列数合并，增多列
print(Z5.shape)'''

5.
'''X = torch.tensor([[1,2,1,2],[1,1,1,1]])
Y = torch.tensor([[1,1,1,1],[1,1,1,1]])
print(X==Y)
print(X.sum())
'''


'''6.
#广播机制
a = torch.arange(4).reshape((4,1))
b = torch.arange(3).reshape((1,3))
print(a,b)
print(a+b)'''


'''7.#访问元素
a = torch.arange(12).reshape(3,4)
print(a)
print(a[-2])
print(a[-1])  #访问最后一行
print(a[1:3])
print(a[0,0])  #访问指定元素
a[0,0] = 99
print(a)
print(a[:,1:2])  #访问第2列元素
a[:1,0:4] = 1     #同时修改多个元素
print(a)'''


'''8.
Y = torch.arange(12)
X = torch.arange(12)
before = id(Y)
Y = Y + X
print(id(Y)==before)  #验证新结果内存地址是否改变
print(id(Y))
#解决方法1
Z = torch.zeros_like(Y)
print('id(Z):',id(Z))
Z[:] = X+Y
print('id(Z):',id(Z))
#解决方法2
before = id(X)
X += Y
print(id(X)==before)'''


'''9.#转换
X = torch.tensor([1.2])
A = X.numpy()
B = torch.tensor(A)
print(type(A),type(B))
print(X.item(),float(X),int(X))'''


'''10.#csv文件转换为张量
os.makedirs(os.path.join('..','data'),exist_ok = True)
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only = True))    
inputs = pd.get_dummies(inputs,dummy_na=True).astype('float32')
print(inputs)
x , y =torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x,y)
'''

#11.线性代数
'''x = torch.arange(9).reshape(3,3)
print(x[1],len(x))
print(x.T)
y = torch.tensor([[1,2,3],[2,1,1],[3,1,2]])
print(y == y.T)     #对称矩阵转置还是本身
z = torch.arange(24).reshape(2,3,4)
print(z)
'''

'''A= torch.arange(20,dtype = torch.float32).reshape(5,4)
B = A
print(id(B) == id(A))
B = A.clone()  #创建张量的副本，返回一个新的张量，与原张量具有相同的数值但不同的内存地址
print(id(B) == id(A))
print(A * B)
'''


'''x = torch.arange(4,dtype = torch.float32)
print(x.sum())
y = torch.arange(12).reshape(2,2,3)
print(y,y.sum())
'''

'''#12.指定求和汇总张量的轴
A = torch.arange(20*2,dtype=torch.float32).reshape(2,5,4)
print(A.shape,A.sum())
print(A)
#有三个维度时
print(A.sum(axis=0))  #按第一维度求和
print(A.sum(axis=1))  #按第二维度求和，保留列
print(A.sum(axis=2))  #按第三维度求和，保留行
print(A.mean(axis=0))
print(A.mean(axis=1))
print(A.shape[0]) #获取第一维度的大小
'''

'''a = torch.ones((2,5,4))
print(a.shape,a.sum().shape)
print(a.sum(axis=[0,2]).shape)
print(a.sum(axis=[0,2],keepdim=True).shape)  #保留去掉的为1
'''

'''
#保持轴数不变
A = torch.arange(20*2,dtype=torch.float32).reshape(2,5,4)
print(A.sum(axis=1,keepdims=True))
print(A.sum(axis=2,keepdims=True))
print(A / A.sum(axis=1,keepdims=True))
print(A / A.sum(axis=2,keepdims=True))
#某个轴计算累积总和
print(A.cumsum(axis=1))
'''

'''#矩阵乘法
x = torch.tensor([1,2,3,4],dtype=torch.float32)
y = torch.ones(4,dtype=torch.float32)
z = torch.arange(12,dtype=torch.float32).reshape(3,4)
print(torch.dot(x,y))
print(torch.sum(x*y))
print(torch.mv(z,y))
#求向量积并将结果拼接一起
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
B = torch.ones(4,3)
print(torch.mm(A,B))
'''

'''#求L2范数（向量元素平方和的平方根）
u = torch.tensor([3.0,-4.0])
print(torch.norm(u))
#求L1范数（向量元素的绝对值之和）
print(torch.abs(u).sum())
#求矩阵的佛罗贝尼乌斯范数
print(torch.norm(torch.ones((4,9))))
'''


'''12.#自动求导
x = torch.arange(4.0)
print(x)
x.requires_grad_(True) #等价于x = torch.arange(4.0,requires_grad=True)
print(x.grad)
y = 2 * torch.dot(x,x)
print(y)
print(y.backward())
print(x.grad)
print(x.grad == 4*x)
'''

'''x.grad.zero_()  #若不清空梯度，则新梯度会加上之前的梯度
y = x.sum()
y.backward()
print(x.grad)'''

'''x.grad.zero_()
y = x*x   #当x*x结果是矩阵时,变成标量求
y.sum().backward()  #等价于y.backward(torch.ones(len(x)))
print(x.grad)'''

'''#分离计算图
x.grad.zero_()
y = x*x
u = y.detach()  #阻断梯度传播，u不再需要梯度
z = u*x
z.sum().backward()
print(x.grad)
print(x.grad == u)'''