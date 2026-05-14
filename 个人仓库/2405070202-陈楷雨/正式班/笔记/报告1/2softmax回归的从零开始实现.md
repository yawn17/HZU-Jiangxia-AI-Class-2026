# softmax回归的从零开始实现

## 1. 导入必要的库并读取数据集

```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 2. 初始化模型参数

将每个 $28 \times 28$ 图像展平为长度为 $784$ 的向量。因为 Fashion-MNIST 有 10 个类别，所以输出维度为 10。

```python
num_inputs = 784
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

## 3. 实现 softmax 运算

softmax 定义为：

$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_{k} \exp(\mathbf{X}_{ik})}
$$

```python
def softmax(X):
	x_exp = torch.exp(X)
	partition = x_exp.sum(1, keepdim=True)  # keepdim=True保持维度不变
	return x_exp / partition  # 这里使用到了广播机制
```

## 4. 验证 softmax 输出

我们希望看到两点：
1. 每个元素都非负。
2. 每一行的概率和为 1。

```python
x = torch.normal(0, 1, (2, 5))
x_prob = softmax(x)
x_prob, x_prob.sum(1)
```

## 5. 定义 softmax 回归模型

```python
def net(X):
	return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)
```

## 6. 取出正确类别对应的预测概率

```python
# 创建一个数据y_hat，其中包含2个样本在3个类别上的预测值，使用y作为y_hat中概率的索引
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

## 7. 实现交叉熵损失函数

```python
def cross_entropy(y_hat, y):
	return -torch.log(y_hat[range(len(y_hat)), y])
cross_entropy(y_hat, y)
```

## 8. 计算分类准确率

```python
def accuracy(y_hat, y):
	"""计算预测正确的数量"""
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
		y_hat = y_hat.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
	return float(cmp.type(y.dtype).sum())
accuracy(y_hat, y) / len(y)
```

## 9. 评估模型在数据集上的准确率

```python
def evaluate_accuracy(net, data_iter):
	"""评估在指定数据集上模型的准确率"""
	if isinstance(net, torch.nn.Module):
		net.eval()  # 将模型设置为评估模式
	metric = d2l.Accumulator(2)  # 正确预测数、预测总数
	with torch.no_grad():
		for X, y in data_iter:
			metric.add(accuracy(net(X), y), y.numel())
	return metric[0] / metric[1]
```

## 10. Accumulator 累加器

```python
class Accumulator:
	"""在n个变量上累加"""
	def __init__(self, n):
		self.data = [0.0] * n

	def add(self, *args):
		self.data = [a + float(b) for a, b in zip(self.data, args)]

	def reset(self):
		self.data = [0.0] * len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
evaluate_accuracy(net, test_iter)
```

## 11. 单轮训练函数

```python
def train_epoch_ch3(net,train_iter,loss,updater):
	if isinstance(net,torch.nn.Module):
		net.train()  # 将模型设置为训练模式
	metric = d2l.Accumulator(3)  # 损失总和、正确预测数、预测总数
	for X,y in train_iter:
		y_hat = net(X)
		l = loss(y_hat,y)
		if isinstance(updater, torch.optim.Optimizer):
			updater.zero_grad()
			l.backward()
			updater.step()
			metric.add(float(l)*len(y),accuracy(y_hat,y),y.size().numel())
		else:
			l.sum().backward()
			updater(X.shape[0])
			metric.add(float(l.sum()),accuracy(y_hat,y),y.size().numel())
	return metric[0]/metric[2], metric[1]/metric[2]
```

## 12. 动画绘图工具

```python
class Animator:  #@save
	"""在动画中绘制数据"""
	def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
				 ylim=None, xscale='linear', yscale='linear',
				 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
				 figsize=(3.5, 2.5)):
		# 增量地绘制多条线
		if legend is None:
			legend = []
		d2l.use_svg_display()
		self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
		if nrows * ncols == 1:
			self.axes = [self.axes, ]
		# 使用lambda函数捕获参数
		self.config_axes = lambda: d2l.set_axes(
			self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
		self.X, self.Y, self.fmts = None, None, fmts

	def add(self, x, y):
		# 向图表中添加多个数据点
		if not hasattr(y, "__len__"):
			y = [y]
		n = len(y)
		if not hasattr(x, "__len__"):
			x = [x] * n
		if not self.X:
			self.X = [[] for _ in range(n)]
		if not self.Y:
			self.Y = [[] for _ in range(n)]
		for i, (a, b) in enumerate(zip(x, y)):
			if a is not None and b is not None:
				self.X[i].append(a)
				self.Y[i].append(b)
		self.axes[0].cla()
		for x, y, fmt in zip(self.X, self.Y, self.fmts):
			self.axes[0].plot(x, y, fmt)
		self.config_axes()
		display.display(self.fig)
		display.clear_output(wait=True)
```

## 13. 完整训练函数

```python
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
						legend=['train loss', 'train acc', 'test acc'])
	for epoch in range(num_epochs):
		train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
		test_acc = evaluate_accuracy(net, test_iter)
		animator.add(epoch + 1, train_metrics + (test_acc,))
	train_loss, train_acc = train_metrics
```

## 14. 定义优化器更新函数

```python
lr = 0.1
def updater(batch_size):
	return d2l.sgd([w, b], lr, batch_size)
```

## 15. 训练模型

```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## 16. 分类预测

```python
def predict_ch3(net, test_iter, n=20):
	for X, y in test_iter:
		break
	trues = d2l.get_fashion_mnist_labels(y)
	preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
	titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
	d2l.show_images(
		X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
predict_ch3(net, test_iter)
```

## 17. 附录：矩阵按维求和示例

```python
# 对矩阵的加运算
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdims=True), X.sum(1, keepdims=True)
```

## 18. 小结

- softmax 回归可以只用张量运算从零实现。
- 关键流程是：定义 softmax、定义交叉熵、计算准确率、按小批量迭代训练。
- 从可视化训练曲线可以同时观察训练损失、训练准确率和测试准确率。

## 思考

1. 从零实现有助于理解概率归一化与交叉熵损失的数学含义。
2. 本节课的目的是为了理解softmax的底层具体运行逻辑，有利于后续对softmax的具体学习。