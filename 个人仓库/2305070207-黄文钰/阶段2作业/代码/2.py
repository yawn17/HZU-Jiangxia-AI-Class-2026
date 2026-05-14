import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR10 彩色32x32图片，10个类别
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 搭建CNN卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层 + 池化
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积+激活+池化
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 32 * 8 * 8)
        # 全连接
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失、优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
def train():
    model.train()
    for epoch in range(5):
        loss_sum = 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"第{epoch+1}轮 | 平均损失: {loss_sum/len(train_loader):.4f}")

# 测试准确率
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            pred = out.argmax(dim=1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    print(f"测试集准确率: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
    test()