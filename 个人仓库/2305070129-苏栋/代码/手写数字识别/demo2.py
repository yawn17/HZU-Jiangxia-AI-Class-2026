import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 解决OpenMP冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
def create_save_dirs():
    base_dir = "activation_curve_mapping"
    subdirs = [
        "input", "conv", "pool", "output", 
        "activation_maps", "activation_curves",  # 新增曲线保存目录
        "activation_visualizations"
    ]
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    # 激活函数子目录
    acts = ["relu", "sigmoid", "tanh", "leaky_relu"]
    for act in acts:
        os.makedirs(os.path.join(base_dir, "activation_maps", act), exist_ok=True)
    return base_dir

# 读取MNIST数据
def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        images = images.copy()
    return images

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# 数据集类
class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx

# 激活函数管理器（含曲线绘制）
class ActivationManager:
    def __init__(self):
        # 激活函数定义
        self.functions = {
            'relu': lambda x: np.maximum(0, x),  # 用numpy实现便于曲线绘制
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.1 * x)
        }
        # PyTorch版本（用于实际计算）
        self.torch_functions = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        # 函数曲线范围（根据池化层输出分布调整）
        self.curve_ranges = {
            'relu': (-2, 6),        # ReLU对负值敏感
            'sigmoid': (-6, 6),     # Sigmoid在±6外趋于饱和
            'tanh': (-3, 3),        # Tanh在±3外趋于饱和
            'leaky_relu': (-4, 4)   # LeakyReLU需展示负值区域
        }
        # 函数描述
        self.descriptions = {
            'relu': 'ReLU: f(x) = max(0, x)',
            'sigmoid': 'Sigmoid: f(x) = 1/(1+e^(-x))',
            'tanh': 'Tanh: f(x) = (e^x - e^(-x))/(e^x + e^(-x))',
            'leaky_relu': 'LeakyReLU: f(x) = x if x>0 else 0.1x'
        }
    
    def get_torch_output(self, x, act_name):
        """用PyTorch计算激活值（用于网络前向传播）"""
        return self.torch_functions[act_name](x)
    
    def plot_activation_curve(self, act_name, input_values, output_values, save_path):
        """绘制激活函数曲线，并标记输入-输出映射点"""
        x_min, x_max = self.curve_ranges[act_name]
        x_curve = np.linspace(x_min, x_max, 1000)  # 曲线x轴
        y_curve = self.functions[act_name](x_curve)  # 曲线y轴（函数值）
        
        # 创建图像
        plt.figure(figsize=(8, 5))
        # 绘制函数曲线
        plt.plot(x_curve, y_curve, 'b-', linewidth=2, label=self.descriptions[act_name])
        # 标记输入-输出映射点（用红色圆点）
        plt.scatter(input_values, output_values, c='red', s=50, zorder=5, 
                   label=f'样本映射点（{len(input_values)}个）')
        # 坐标轴与网格
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(alpha=0.3)
        plt.xlabel('输入值（池化层输出）')
        plt.ylabel('输出值（激活后）')
        plt.title(f'{act_name} 激活函数曲线与样本映射点')
        plt.legend()
        plt.tight_layout()
        # 保存图像
        plt.savefig(save_path)
        plt.close()

# 卷积神经网络
class CNNWithCurveMapping(nn.Module):
    def __init__(self):
        super(CNNWithCurveMapping, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.act_manager = ActivationManager()
        self.fc = nn.Linear(16 * 14 * 14, 10)
        self.layer_outputs = {}
        self.activation_maps = {}  # 激活函数输出（特征图）
    
    def forward(self, x):
        self.layer_outputs['input'] = x
        x_conv = self.conv1(x)
        self.layer_outputs['conv'] = x_conv
        x_pool = self.pool(x_conv)
        self.layer_outputs['pool'] = x_pool
        
        # 计算所有激活函数的特征图
        self.activation_maps = {
            name: self.act_manager.get_torch_output(x_pool, name) 
            for name in self.act_manager.functions.keys()
        }
        
        # 主激活函数（ReLU）用于分类
        x_act = self.act_manager.torch_functions['relu'](x_pool)
        self.layer_outputs['output'] = self.fc(x_act.view(-1, 16*14*14))
        return self.layer_outputs['output']

# 提取激活函数的输入输出样本（用于曲线映射）
def get_activation_samples(pool_output, activation_map, num_samples=20):
    """从池化层输出（激活输入）和激活输出中提取样本点"""
    # 转换为numpy数组（展平为1D）
    pool_flat = pool_output.squeeze().cpu().numpy().flatten()  # 激活函数输入
    act_flat = activation_map.squeeze().cpu().numpy().flatten()  # 激活函数输出
    
    # 随机采样（避免样本过多导致曲线杂乱）
    if len(pool_flat) > num_samples:
        indices = np.random.choice(len(pool_flat), num_samples, replace=False)
        return pool_flat[indices], act_flat[indices]
    return pool_flat, act_flat

# 保存激活函数曲线映射结果
def save_activation_curve_mapping(base_dir, sample_idx, pool_output, activation_maps):
    """为每种激活函数绘制曲线映射图"""
    for act_name, act_map in activation_maps.items():
        # 提取输入（池化层输出）和输出（激活后）的样本点
        input_samples, output_samples = get_activation_samples(pool_output, act_map)
        
        # 绘制并保存曲线映射图
        save_path = os.path.join(
            base_dir, "activation_curves", 
            f"sample_{sample_idx}_{act_name}_mapping.png"
        )
        activation_manager = ActivationManager()  # 复用激活函数定义
        activation_manager.plot_activation_curve(
            act_name=act_name,
            input_values=input_samples,
            output_values=output_samples,
            save_path=save_path
        )
    return input_samples  # 返回一组样本用于后续打印

# 综合处理与可视化
def process_sample(model, image, label, sample_idx, base_dir):
    model.eval()
    with torch.no_grad():
        model(image.unsqueeze(0))  # 前向传播获取所有层输出
    
    # 保存基础层数据
    input_np = model.layer_outputs['input'].squeeze().cpu().numpy()
    np.save(os.path.join(base_dir, "input", f"sample_{sample_idx}.npy"), input_np)
    plt.imsave(os.path.join(base_dir, "input", f"sample_{sample_idx}.png"), input_np, cmap='gray')
    
    # 保存激活函数特征图
    for act_name, act_map in model.activation_maps.items():
        act_np = act_map.squeeze().cpu().numpy()
        np.save(os.path.join(base_dir, "activation_maps", act_name, f"sample_{sample_idx}.npy"), act_np)
    
    # 绘制并保存激活函数曲线映射（核心功能）
    pool_output = model.layer_outputs['pool']  # 激活函数的输入
    input_samples = save_activation_curve_mapping(base_dir, sample_idx, pool_output, model.activation_maps)
    
    # 打印映射示例（输入→输出）
    print(f"\n===== 样本 {sample_idx} 激活函数映射示例 =====")
    print(f"从池化层输出中采样 {len(input_samples)} 个值，展示映射关系：")
    for i, x in enumerate(input_samples[:5]):  # 展示前5个
        print(f"输入值: {x:.4f} → "
              f"ReLU输出: {model.act_manager.functions['relu'](x):.4f} | "
              f"Sigmoid输出: {model.act_manager.functions['sigmoid'](x):.4f} | "
              f"Tanh输出: {model.act_manager.functions['tanh'](x):.4f}")
    print("============================================\n")

def main():
    base_dir = create_save_dirs()
    print(f"结果保存至: {base_dir}")
    
    # 读取数据
    print("读取MNIST数据集...")
    train_images = read_mnist_images('数据集/train-images.idx3-ubyte')
    train_labels = read_mnist_labels('数据集/train-labels.idx1-ubyte')
    print(f"训练集规模: {train_images.shape}, 标签: {train_labels.shape}")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 数据加载
    dataset = MNISTDataset(train_images, train_labels, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 模型与训练
    model = CNNWithCurveMapping()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 2
    print(f"\n开始训练（{num_epochs}个epoch）...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels, _) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(loader)}], Loss: {running_loss/200:.4f}')
                running_loss = 0.0
        
        # 处理样本并生成激活函数曲线映射
        print(f"\nEpoch {epoch+1} 完成，生成激活函数曲线映射...")
        model.eval()
        with torch.no_grad():
            sample_indices = [100 + epoch*2, 200 + epoch*2]  # 选择样本
            for idx in sample_indices:
                if idx < len(dataset):
                    image, label, sample_idx = dataset[idx]
                    process_sample(model, image, label, sample_idx, base_dir)
    
    print("\n所有样本处理完成！激活函数曲线映射结果已保存至 activation_curves 目录。")

if __name__ == "__main__":
    main()