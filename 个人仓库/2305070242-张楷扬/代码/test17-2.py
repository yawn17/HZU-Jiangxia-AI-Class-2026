import torch
''''
x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
print(x.grad)

y = 2*torch.dot(x, x)
print(y)

y.backward()
print(x.grad)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)
'''
def f(a):
    b = a*2
    while b.norm()< 1000:
        b = b*2
    if b.sum()> 0:
        c = b
    else:
        c = 100*b
    return c
a = torch.randn(size=(),requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d/a)
