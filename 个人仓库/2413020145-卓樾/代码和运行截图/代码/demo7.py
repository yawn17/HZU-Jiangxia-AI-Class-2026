import torch

x = torch.arange(4.0, requires_grad=True)

for i in range(50):
    if x.grad is not None:
        x.grad.zero_()
    y = 2 * torch.dot(x, x)
    y.backward()
    if (i + 1) % 10 == 0:

        print(f"第{i + 1}次 - 梯度: {x.grad}")

print("\n梯度验证: x.grad == 4*x →", x.grad.equal(4 * x))