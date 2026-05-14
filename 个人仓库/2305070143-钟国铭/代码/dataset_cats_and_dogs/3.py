import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from collections import defaultdict

# 设置Matplotlib字体以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用微软雅黑字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def simple_pca(data, n_components=2):
    mean = np.mean(data, axis=0)
    centered = data - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    projected = np.dot(centered, components.T)
    return projected

def visualize_activation_distributions_per_class(model, val_loader, device, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    activations_cat = defaultdict(list)
    activations_dog = defaultdict(list)
    features_mixed = defaultdict(list)
    labels_mixed = []
    activation_hooks = {}

    def get_activation(name):
        def hook(module, input, output):
            activation_hooks[name] = output.detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(get_activation(name)))

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _ = model(inputs)
            for name, activation in activation_hooks.items():
                activation = activation.cpu().numpy()
                num_batch = activation.shape[0]
                flat_dim = np.prod(activation.shape[1:])
                flat_activation = activation.reshape(num_batch, flat_dim)
                cat_mask = (labels == 0).cpu().numpy()
                dog_mask = (labels == 1).cpu().numpy()
                if np.any(cat_mask):
                    activations_cat[name].append(flat_activation[cat_mask])
                if np.any(dog_mask):
                    activations_dog[name].append(flat_activation[dog_mask])
                features_mixed[name].append(flat_activation)
            labels_mixed.append(labels.cpu().numpy())

    for hook in hooks:
        hook.remove()

    for name in activations_cat:
        activations_cat[name] = np.concatenate(activations_cat[name], axis=0) if activations_cat[name] else np.array([])
        activations_dog[name] = np.concatenate(activations_dog[name], axis=0) if activations_dog[name] else np.array([])

    features_mixed_concat = {}
    for name in features_mixed:
        features_mixed_concat[name] = np.concatenate(features_mixed[name], axis=0)

    labels_mixed = np.concatenate(labels_mixed)

    categories = {
        'cat': (activations_cat, np.zeros(activations_cat[list(activations_cat.keys())[0]].shape[0] if activations_cat else 0)),
        'dog': (activations_dog, np.ones(activations_dog[list(activations_dog.keys())[0]].shape[0] if activations_dog else 0)),
        'mixed': (features_mixed_concat, labels_mixed)
    }

    for cat_name, (features_dict, cat_labels) in categories.items():
        for layer_name, features in features_dict.items():
            if features.shape[0] == 0:
                continue
            # 分布直方图
            flat_values = features.flatten()
            plt.figure(figsize=(10, 5))
            plt.hist(flat_values, bins=50, alpha=0.7, color='blue', density=True)
            plt.title(f'{layer_name} 激活值分布 - {cat_name}')
            plt.xlabel('激活值')
            plt.ylabel('密度')
            save_path = os.path.join(save_dir, f'activation_hist_{cat_name}_{layer_name}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"保存激活函数分布图像: {save_path}")

            # 2D投影观察
            if features.shape[0] > 2 and features.shape[1] > 2:
                try:
                    projected = simple_pca(features)
                    plt.figure(figsize=(10, 5))
                    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=cat_labels, cmap='coolwarm', alpha=0.5)
                    plt.colorbar(scatter, ticks=[0, 1], format=plt.FuncFormatter(lambda val, loc: ['猫', '狗'][int(val)]))
                    plt.title(f'{layer_name} 激活值2D投影 - {cat_name}')
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    save_path = os.path.join(save_dir, f'activation_2d_{cat_name}_{layer_name}.png')
                    plt.savefig(save_path)
                    plt.close()
                    print(f"保存2D投影图像: {save_path}")
                except Exception as e:
                    print(f"{layer_name} {cat_name} 的PCA失败: {e}")

def load_data():
    print("="*60)
    print("步骤1: 数据加载")
    print("="*60)
    
    class CatDogDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.samples = []
            self.root_dir = root_dir
            self.transform = transform
            
            with open(csv_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            path, label = parts[0].strip(), int(parts[1].strip())
                            self.samples.append((path, label))
            
            print(f"加载 {len(self.samples)} 个样本")
        
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            full_path = os.path.join(self.root_dir, path)
            try:
                image = Image.open(full_path).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), color='black')
            if self.transform: image = self.transform(image)
            return image, label
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    base_dir = r'C:\Users\ZGM\Desktop\python\torch_deeplearning\dataset_cats_and_dogs'
    train_dataset = CatDogDataset(os.path.join(base_dir, 'train.csv'), base_dir, data_transforms['train'])
    val_dataset = CatDogDataset(os.path.join(base_dir, 'val.csv'), base_dir, data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    train_labels = [label for _, label in train_dataset.samples]
    val_labels = [label for _, label in val_dataset.samples]
    train_counts = [train_labels.count(0), train_labels.count(1)]
    val_counts = [val_labels.count(0), val_labels.count(1)]
    
    print(f"训练集: 猫 {train_counts[0]}张, 狗 {train_counts[1]}张")
    print(f"验证集: 猫 {val_counts[0]}张, 狗 {val_counts[1]}张")
    
    return train_loader, val_loader, train_dataset, val_dataset

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, return_prob=False):
        logits = self.classifier(self.conv_layers(x))
        
        if return_prob:
            probs = self.softmax(logits)
            return logits, probs
        
        return logits

class ActivationCatcher:
    def __init__(self):
        self.activations = []
        self.hooks = []
        self.logits = []
    
    def register_hooks(self, model):
        def get_activation_hook(name):
            def hook(module, input, output):
                if hasattr(self, 'capture_flag') and self.capture_flag:
                    activation = output.detach().cpu().numpy()
                    self.activations.append((name, activation))
            return hook
        
        self.remove_hooks()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(hook)
    
    def get_activations(self):
        return self.activations
    
    def clear_activations(self):
        self.activations = []
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_activations(self):
        if not self.activations:
            return None
        
        stats = []
        for name, activation in self.activations:
            flat_activation = activation.flatten()
            mean_val = np.mean(flat_activation)
            std_val = np.std(flat_activation)
            min_val = np.min(flat_activation)
            max_val = np.max(flat_activation)
            zero_percent = np.mean(flat_activation == 0) * 100
            
            stats.append({
                'layer': name,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'zero_percent': zero_percent
            })
        
        return stats
    
    def visualize_activations(self):
        if not self.activations:
            print("没有可可视化的激活值数据")
            return
        
        plt.figure(figsize=(15, len(self.activations) * 4))
        
        for i, (layer_name, activation) in enumerate(self.activations, 1):
            all_activations = activation
            flat_activations = all_activations.flatten()
            
            plt.subplot(len(self.activations), 1, i)
            
            plt.hist(flat_activations, bins=50, alpha=0.7, color='blue', density=True)
            plt.title(f'{layer_name} 激活值分布')
            plt.xlabel('激活值')
            plt.ylabel('频率')
            
            stats_text = f'均值: {np.mean(flat_activations):.4f}\n'
            stats_text += f'标准差: {np.std(flat_activations):.4f}\n'
            stats_text += f'零值比例: {(np.sum(flat_activations == 0) / len(flat_activations)) * 100:.2f}%'
            plt.figtext(0.91, 0.5, stats_text, verticalalignment='center', 
                       transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('relu_activation_distributions.png')
        plt.close()
        print("激活值分布图已保存为 'relu_activation_distributions.png'")
    
    def analyze_logits(self):
        if not self.logits:
            print("没有捕获到logits数据")
            return None
        
        all_logits = np.concatenate(self.logits, axis=0)
        
        class0_logits = all_logits[:, 0]
        class1_logits = all_logits[:, 1]
        
        results = {
            'class0': {
                'mean': np.mean(class0_logits),
                'std': np.std(class0_logits),
                'min': np.min(class0_logits),
                'max': np.max(class0_logits)
            },
            'class1': {
                'mean': np.mean(class1_logits),
                'std': np.std(class1_logits),
                'min': np.min(class1_logits),
                'max': np.max(class1_logits)
            },
            'magnitude_stats': {
                'mean_absolute': np.mean(np.abs(all_logits)),
                'mean_diff': np.mean(np.abs(class0_logits - class1_logits)),
                'large_diff_percent': (np.sum(np.abs(class0_logits - class1_logits) > 5) / len(all_logits)) * 100
            }
        }
        
        return results
    
    def visualize_logits(self):
        if not self.logits:
            print("没有可可视化的logits数据")
            return
        
        all_logits = np.concatenate(self.logits, axis=0)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(all_logits[:, 0], bins=50, alpha=0.5, color='blue', label='类别0 (Cat)')
        plt.hist(all_logits[:, 1], bins=50, alpha=0.5, color='red', label='类别1 (Dog)')
        plt.title('各类别Logits分布')
        plt.xlabel('Logit值')
        plt.ylabel('频率')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.scatter(all_logits[:, 0], all_logits[:, 1], alpha=0.5)
        plt.plot([np.min(all_logits), np.max(all_logits)], 
                [np.min(all_logits), np.max(all_logits)], 'k--', alpha=0.5)
        plt.title('类别0 vs 类别1 Logits')
        plt.xlabel('类别0 Logit')
        plt.ylabel('类别1 Logit')
        
        plt.subplot(2, 2, 3)
        logits_diff = all_logits[:, 0] - all_logits[:, 1]
        plt.hist(logits_diff, bins=50, alpha=0.7, color='green')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Logits差值分布（类别0 - 类别1）')
        plt.xlabel('Logit差值')
        plt.ylabel('频率')
        
        plt.subplot(2, 2, 4)
        plt.hist(np.abs(all_logits.flatten()), bins=50, alpha=0.7, color='purple')
        plt.title('Logits绝对值分布')
        plt.xlabel('Logit绝对值')
        plt.ylabel('频率')
        
        plt.tight_layout()
        plt.savefig('softmax_input_logits_distribution.png')
        plt.close()
        print("Softmax输入（logits）分布图已保存为 'softmax_input_logits_distribution.png'")

def build_model():
    print("\n" + "="*60)
    print("步骤2: 模型构建")
    print("="*60)
    
    model = CatDogCNN()
    
    print("模型结构:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=25, analyze_activations=True, patience=5):
    print("\n" + "="*60)
    print("步骤3: 模型训练")
    print("="*60)
    print(f"训练配置: {num_epochs}个epoch, 早停耐心值: {patience}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("\n计算类别权重来缓解数据不平衡...")
    class_counts = torch.zeros(2).to(device)
    for _, labels in train_loader:
        labels = labels.to(device)
        for label in labels:
            class_counts[label] += 1
    
    print(f"类别分布 - 猫: {class_counts[0].item()}, 狗: {class_counts[1].item()}")
    
    total_samples = class_counts.sum()
    class_weights = total_samples / (2.0 * class_counts)
    print(f"类别权重 - 猫: {class_weights[0].item():.4f}, 狗: {class_weights[1].item():.4f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_epoch = 0
    
    activation_catcher = ActivationCatcher()
    if analyze_activations:
        activation_catcher.register_hooks(model)
        activation_catcher.capture_flag = False
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if analyze_activations and epoch == 0 and batch_idx < 3:
                activation_catcher.capture_flag = True
                activation_catcher.clear_activations()
            else:
                activation_catcher.capture_flag = False
            
            logits = model(inputs)
            
            if analyze_activations and epoch == 0 and batch_idx < 3:
                activation_catcher.logits.append(logits.detach().cpu().numpy())
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
        
        model.eval()
        val_loss, val_correct = 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                logits, probs = model(inputs, return_prob=True)
                
                val_loss += criterion(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch + 1
            print(f"  验证准确率提升: {val_acc:.4f}, 保存最佳模型权重")
        else:
            patience_counter += 1
            print(f"  验证准确率未提升, 耐心计数器: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"早停触发! 最佳验证准确率: {best_val_acc:.4f} 在 Epoch {best_epoch}")
                model.load_state_dict(best_model_weights)
                break
        
        print(f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
        print(f'训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}')
        print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if analyze_activations and epoch == 0:
            print("\n===== ReLU激活值分析 =====")
            stats = activation_catcher.analyze_activations()
            if stats:
                for stat in stats:
                    print(f"层: {stat['layer']}")
                    print(f"  均值: {stat['mean']:.4f}")
                    print(f"  标准差: {stat['std']:.4f}")
                    print(f"  最小值: {stat['min']:.4f}")
                    print(f"  最大值: {stat['max']:.4f}")
                    print(f"  零值比例: {stat['zero_percent']:.2f}%")
                    print()
                activation_catcher.visualize_activations()
            else:
                print("未捕获到激活值")
            print("=========================\n")
            
            print("\n===== Softmax输入（logits）分析 =====")
            logits_stats = activation_catcher.analyze_logits()
            if logits_stats:
                print(f"类别0 (Cat) 统计:")
                print(f"  均值: {logits_stats['class0']['mean']:.4f}")
                print(f"  标准差: {logits_stats['class0']['std']:.4f}")
                print(f"  范围: [{logits_stats['class0']['min']:.4f}, {logits_stats['class0']['max']:.4f}]")
                print()
                print(f"类别1 (Dog) 统计:")
                print(f"  均值: {logits_stats['class1']['mean']:.4f}")
                print(f"  标准差: {logits_stats['class1']['std']:.4f}")
                print(f"  范围: [{logits_stats['class1']['min']:.4f}, {logits_stats['class1']['max']:.4f}]")
                print()
                print(f"幅度统计:")
                print(f"  平均绝对值: {logits_stats['magnitude_stats']['mean_absolute']:.4f}")
                print(f"  平均类别差值: {logits_stats['magnitude_stats']['mean_diff']:.4f}")
                print(f"  大差值比例 (|diff| > 5): {logits_stats['magnitude_stats']['large_diff_percent']:.2f}%")
                activation_catcher.visualize_logits()
            else:
                print("未捕获到logits数据")
            print("==================================\n")
        
        scheduler.step(val_loss)
    
    model.load_state_dict(best_model_weights)
    
    print("\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f} 在 Epoch {best_epoch}")
    
    results = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }
    
    return results

def evaluate_model(model, val_loader):
    print("\n" + "="*60)
    print("步骤4: 模型评估")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.hstack(all_preds)
    all_labels = np.hstack(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    precision = np.sum((all_preds == 1) & (all_labels == 1)) / np.sum(all_preds == 1) if np.sum(all_preds == 1) > 0 else 0
    recall = np.sum((all_preds == 1) & (all_labels == 1)) / np.sum(all_labels == 1) if np.sum(all_labels == 1) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return accuracy, precision, recall, f1

def main():
    print("开始神经网络训练...")
    
    train_loader, val_loader, train_dataset, val_dataset = load_data()
    
    model = build_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for round in range(3):
        print(f"\n多轮自适应优化 - 轮次 {round+1}")
        results = train_model(model, train_loader, val_loader, num_epochs=10, patience=3, analyze_activations=(round==0))
        model = results['model']
        visualize_activation_distributions_per_class(model, val_loader, device)
    
    evaluate_model(results['model'], val_loader)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == '__main__':
    main()