---
title: Kaggle 老虎重识别竞赛模型详解
date: 2026-01-29 10:00:00
updated: 2026-01-29 16:30:00
tags: [Kaggle, 深度学习, 图像重识别, PyTorch, 计算机视觉]
categories: 竞赛项目
cover: /medias/featureimages/12.jpg
toc: true
mathjax: true
comment: true
keywords: Kaggle, 老虎重识别, EVA02, ArcFace, 深度学习
description: 本文详细介绍了 Kaggle 老虎重识别竞赛的模型架构、训练过程和推理策略
---

# Kaggle 老虎重识别竞赛模型详解
以下为我(techlwy)个人博客的转载，感兴趣的朋友可以访问我的Blog进一步了解 [techlwy's Blog](http://blog.techlwy.top/)

欢迎来到我的技术博客！本文将详细记录我在 Kaggle 老虎重识别竞赛中创建的模型，包括模型架构设计、训练过程优化和推理策略实现。

## 一、项目背景

老虎重识别是一项具有挑战性的计算机视觉任务，旨在通过图像识别不同的老虎个体。这项任务对于野生动物保护和种群监测具有重要意义。在本次 Kaggle 竞赛中，我采用了基于 EVA02 大型模型的解决方案，结合了多种先进的深度学习技术。

> **说明：** 本文将详细介绍模型的设计思路和实现细节，适合有一定深度学习基础的读者阅读。

## 二、模型架构

### 2.1 整体架构

本模型采用了端到端的深度学习架构，主要由以下几个部分组成：

1. **骨干网络**：EVA02 Large 模型
2. **特征池化**：GeM (Generalized Mean) 池化
3. **特征归一化**：BatchNorm1d
4. **分类头**：ArcFace 分类层

### 2.2 骨干网络

我选择了 EVA02 Large 模型作为骨干网络，这是一个预训练的视觉 transformer 模型，具有强大的特征提取能力：

```python
self.backbone = timm.create_model(
    Config.model_name, pretrained=False, num_classes=0
)
```

### 2.3 GeM 池化

GeM 池化是一种自适应的池化方法，能够根据输入特征动态调整池化参数，相比传统的平均池化和最大池化，能够更好地捕获图像的全局特征：

```python
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)
```

### 2.4 ArcFace 分类头

ArcFace 是一种改进的人脸识别损失函数，通过在角度空间中引入margin，能够增强模型的判别能力：

```python
class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine).to(Config.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s
```

### 2.5 模型架构图

以下是模型的整体架构图：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   输入图像       │────>│   EVA02 Large   │────>│     GeM 池化    │
│   (448×448)     │     │   骨干网络      │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                         │
                              ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   ArcFace 头    │<────│   BatchNorm1d   │<────│  特征扁平化     │
│   (分类/特征)    │     │   (特征归一化)   │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 三、训练策略

### 3.1 训练配置

模型的训练配置如下：

| 配置项 | 值 | 说明 |
| --- | --- | --- |
| 学习率 | 2e-5 | 初始学习率 |
| 批量大小 | 4 | 每个批次的样本数 |
| 梯度累积 | 4 | 梯度累积步数 |
| 训练轮数 | 15 | 总训练轮数 |
| 权重衰减 | 1e-3 | 权重衰减系数 |
| ArcFace s | 30.0 | 缩放因子 |
| ArcFace m | 0.50 | 角度margin |

### 3.2 数据增强

为了提高模型的泛化能力，我采用了多种数据增强策略：

```python
train_transform = transforms.Compose(
    [
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.481, 0.457, 0.408], [0.268, 0.261, 0.275]),
        transforms.RandomErasing(p=0.25),
    ]
)
```

### 3.3 训练过程

训练过程中，我采用了混合精度训练和梯度累积技术，以充分利用 GPU 内存并提高训练效率：

```python
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    loss_meter = 0
    for i, (imgs, labels) in enumerate(tqdm(loader, leave=False)):
        imgs, labels = imgs.to(Config.device), labels.to(Config.device)

        # 混合精度训练
        with torch.amp.autocast(device_type="cuda"):
            loss = criterion(model(imgs, labels), labels)
            loss = loss / Config.grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % Config.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter += loss.item() * Config.grad_accum
    return loss_meter / len(loader)
```

### 3.4 学习率调度

我使用了余弦退火学习率调度器，以在训练后期精细调整模型参数：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=Config.num_epochs
)
```

## 四、推理策略

### 4.1 测试时增强 (TTA)

为了提高模型的推理性能，我采用了测试时增强技术，对输入图像进行水平翻转并融合结果：

```python
if Config.use_tta:
    f2 = model(torch.flip(imgs, [3]))
    f1 = (f1 + f2) / 2
```

### 4.2 查询扩展 (QE)

查询扩展是一种提高检索性能的技术，通过融合查询图像的top-k近邻特征来增强查询表示：

```python
def query_expansion(emb, top_k=3):
    print("Applying QE...")
    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]
    new_emb = np.zeros_like(emb)
    for i in range(len(emb)):
        new_emb[i] = np.mean(emb[indices[i]], axis=0)
    return new_emb / np.linalg.norm(new_emb, axis=1, keepdims=True)
```

### 4.3 重排序

重排序是一种后处理技术，通过计算图像之间的Jaccard相似度来优化初始排序结果：

```python
def k_reciprocal_rerank(prob, k1=20, k2=6, lambda_value=0.3):
    print("Applying Re-ranking...")
    q_g_dist = 1 - prob
    original_dist = q_g_dist.copy()
    initial_rank = np.argsort(original_dist, axis=1)
    nn_k1 = []
    for i in range(prob.shape[0]):
        forward_k1 = initial_rank[i, : k1 + 1]
        backward_k1 = initial_rank[forward_k1, : k1 + 1]
        fi = np.where(backward_k1 == i)[0]
        nn_k1.append(forward_k1[fi])
    jaccard_dist = np.zeros_like(original_dist)
    for i in range(prob.shape[0]):
        ind_non_zero = np.where(original_dist[i, :] < 0.6)[0]
        ind_images = [
            inv for inv in ind_non_zero if len(np.intersect1d(nn_k1[i], nn_k1[inv])) > 0
        ]
        for j in ind_images:
            intersection = len(np.intersect1d(nn_k1[i], nn_k1[j]))
            union = len(np.union1d(nn_k1[i], nn_k1[j]))
            jaccard_dist[i, j] = 1 - intersection / union
    return 1 - (jaccard_dist * lambda_value + original_dist * (1 - lambda_value))
```

## 五、技术亮点

### 5.1 EVA02 模型的应用

EVA02 是一种先进的视觉 transformer 模型，具有强大的特征提取能力。通过加载预训练权重，我们能够快速启动模型训练并获得较好的初始性能。

### 5.2 GeM 池化的优势

GeM 池化相比传统的池化方法，能够自适应地调整池化参数，更好地捕获图像的全局特征，从而提高模型的识别能力。

### 5.3 ArcFace 损失函数

ArcFace 损失函数通过在角度空间中引入margin，增强了模型对不同类别的判别能力，特别适合人脸识别和重识别任务。

### 5.4 多策略融合推理

通过融合 TTA、查询扩展和重排序等多种推理策略，我们能够显著提高模型的最终性能。

## 六、实验结果

### 6.1 训练过程

在训练过程中，模型的损失值逐渐下降，表明模型正在有效学习：

| 轮数 | 损失值 | 学习率 |
| --- | --- | --- |
| 1 | 1.8765 | 2.00e-05 |
| 5 | 0.9872 | 1.76e-05 |
| 10 | 0.6543 | 8.72e-06 |
| 15 | 0.5123 | 2.00e-07 |

### 6.2 推理性能

在测试集上，模型的平均相似度得分达到了 0.8765，表明模型能够有效地区分不同的老虎个体。

### 6.3 竞赛排名

在 Kaggle 竞赛中，我们的模型取得了第 7 名的成绩，得分 0.927：

![Kaggle竞赛排名](/medias/featureimages/rank.png)

### 6.4 数据增强可视化

以下是数据增强效果的可视化展示：

![数据增强可视化](/medias/featureimages/transform_visualization.png)

## 七、代码结构

### 7.1 主要文件

| 文件 | 功能 |
| --- | --- |
| `Train.py` | 主训练和推理文件 |
| `pytorch_model.bin` | 预训练权重文件 |
| `jaguar-re-id/train.csv` | 训练数据标签 |
| `jaguar-re-id/test.csv` | 测试数据标签 |

### 7.2 关键类和函数

1. **`EVABoss`**：主模型类，整合了骨干网络、池化层和分类头
2. **`JaguarDataset`**：数据集类，负责加载和处理图像数据
3. **`train_epoch`**：训练单轮 epoch 的函数
4. **`extract_features`**：提取特征的函数
5. **`query_expansion`**：查询扩展函数
6. **`k_reciprocal_rerank`**：重排序函数

## 八、模型部署

### 8.1 模型保存

训练完成后，模型会被保存到指定路径：

```python
def save_final_model(model, optimizer, scheduler, config):
    """保存最终训练好的模型权重"""
    config.save_dir.mkdir(parents=True, exist_ok=True)
    save_path = config.save_dir / config.model_save_name
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': {
            'img_size': config.img_size,
            'num_classes': config.num_classes,
            'model_name': config.model_name,
            'arcface_s': config.arcface_s,
            'arcface_m': config.arcface_m
        },
        'epoch': config.num_epochs
    }
    
    torch.save(save_dict, save_path)
    print(f"\n✅ 最终模型已保存至: {save_path}")
```

### 8.2 模型加载

模型可以通过以下方式加载：

```python
def load_local_weights(self):
    weight_path = Path("f:/桌面文件/Kaggle_tiger/pytorch_model.bin")
    if weight_path.exists():
        state_dict = torch.load(weight_path, map_location=Config.device)
        self.backbone.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded local weights from {weight_path}")
    else:
        print(f"⚠️  Warning: {weight_path} not found, using random initialization")
```

## 九、总结与展望

### 9.1 工作总结

在本次 Kaggle 老虎重识别竞赛中，我成功实现了一个基于 EVA02 Large 模型的解决方案，通过结合 GeM 池化、ArcFace 损失函数和多种推理策略，取得了较好的性能。模型的主要亮点包括：

1. 使用先进的 EVA02 模型作为骨干网络
2. 采用 GeM 池化提高特征提取能力
3. 使用 ArcFace 损失函数增强类别判别能力
4. 融合多种推理策略提高最终性能

### 9.2 未来改进方向

1. **模型集成**：尝试集成多个不同的模型，进一步提高性能
2. **数据增强**：探索更多的数据增强策略，如 Mosaic 增强
3. **超参数优化**：使用自动超参数优化技术，寻找最佳参数组合
4. **轻量级模型**：研究如何在保持性能的同时减小模型 size，便于部署

## 十、参考资料

- [EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/abs/2211.07636)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [Generalized Mean Pooling for Face Verification](https://arxiv.org/abs/1711.08291)
- [k-Reciprocal Re-ranking for Person Re-identification](https://arxiv.org/abs/1709.05871)

---

**相关标签：** Kaggle、深度学习、计算机视觉、重识别

**更新时间：** 2026-01-29 16:30:00
