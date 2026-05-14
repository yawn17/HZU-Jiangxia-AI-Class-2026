# softmax回归的简洁实现

## 1. 导入必要的库并读取数据集

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 2. 定义模型网络结构

使用 PyTorch 的高级 API `nn.Sequential` 构建网络：
- `Flatten` 层将输入图像展平为一维向量（28×28 → 784）
- `Linear` 层实现全连接，输出维度为 10（对应 10 个类别）

```python
# pytorch不会隐式地调整输入的形状，因此我们需要在这里调整输入的形状
# 我们定义展平层（flatten），在线性层前调整网络输入的形状
# flatten就是将任何维度的展开为2d
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
```

## 3. 初始化模型权重

使用正态分布初始化线性层的权重，标准差设为 0.01。

```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)
```

## 4. 定义损失函数

使用 PyTorch 提供的交叉熵损失函数 `CrossEntropyLoss`。该函数内部会计算 softmax 和对数，因此我们只需传递未归一化的预测值。

```python
loss = nn.CrossEntropyLoss()
```

## 5. 定义优化算法

使用学习率为 0.1 的小批量随机梯度下降（SGD）作为优化算法。

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

## 6. 定义评估和训练函数

### 6.1 计算分类准确率

```python
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(torch.float32).sum())
```

### 6.2 评估模型在数据集上的准确率

```python
def evaluate_accuracy(net, data_iter):
    net.eval()
    device = next(net.parameters()).device
    metric_correct, metric_total = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            metric_correct += accuracy(y_hat, y)
            metric_total += y.numel()
    return metric_correct / metric_total
```

### 6.3 训练函数

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer):
    device = next(net.parameters()).device
    for epoch in range(num_epochs):
        net.train()
        total_loss, total_correct, total_num = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()

            total_loss += float(l) * y.numel()
            total_correct += accuracy(y_hat, y)
            total_num += y.numel()

        train_loss = total_loss / total_num
        train_acc = total_correct / total_num
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch {epoch+1}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
```

## 7. 训练模型

设置训练轮数为 10，开始训练。

```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 8. 训练结果

```
epoch 1: loss=0.7933, train_acc=0.7434, test_acc=0.7839
epoch 2: loss=0.5771, train_acc=0.8095, test_acc=0.8061
epoch 3: loss=0.5308, train_acc=0.8227, test_acc=0.7937
epoch 4: loss=0.5047, train_acc=0.8309, test_acc=0.8107
epoch 5: loss=0.4879, train_acc=0.8356, test_acc=0.8259
epoch 6: loss=0.4760, train_acc=0.8392, test_acc=0.8280
epoch 7: loss=0.4670, train_acc=0.8415, test_acc=0.8319
epoch 8: loss=0.4603, train_acc=0.8444, test_acc=0.8261
epoch 9: loss=0.4538, train_acc=0.8460, test_acc=0.8228
epoch 10: loss=0.4482, train_acc=0.8487, test_acc=0.8319
```

## 9. 小结

- 使用 PyTorch 高级 API 可以更简洁地实现 softmax 回归模型。
- `nn.Sequential` 容器可以方便地堆叠网络层。
- `nn.CrossEntropyLoss` 集成了 softmax 归一化和交叉熵计算，简化了代码。
- 训练过程与从零实现相同，但代码更加清晰、易于维护。
- 最终模型在测试集上达到约 83% 的准确率。

## 思考

1. 简洁实现利用了深度学习框架的高级抽象，减少了手动实现细节的工作量
2. 通过框架内置的优化算法和损失函数，可以更好的去调节参数
3. 相比从零实现，简洁实现的代码更少、更易读，适合进行训练，直接使用现成的模板