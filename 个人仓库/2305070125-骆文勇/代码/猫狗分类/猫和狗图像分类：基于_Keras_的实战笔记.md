# 猫和狗图像分类：基于 Keras 的实战笔记

# 猫狗图像分类实战笔记（从0构建模型）

本次实战基于Keras官方示例，实现从磁盘JPEG文件入手的猫狗二分类任务，全程不使用预训练权重，完整走通**数据预处理-模型构建-训练-推理**的计算机视觉分类流程，同时掌握Keras数据增强、数据集优化、自定义模型架构的核心用法。

## 一、前期准备与数据处理

### 1. 环境与数据加载

核心依赖：`keras`、`tensorflow.data`、`numpy`、`matplotlib`，数据集采用Kaggle猫狗二分类数据集，包含Cat和Dog两个子文件夹的图像文件。

通过`keras.utils.image_dataset_from_directory`直接从目录生成数据集，同时完成**验证集划分**、**图像尺寸统一**、**批次设置**，这是Keras处理图像数据集的高效方式：

```Python

image_size = (180, 180)  # 统一图像尺寸
batch_size = 128
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,  # 按8:2划分训练/验证集
    subset="both",
    seed=1337,  # 固定随机种子保证可复现
    image_size=image_size,
    batch_size=batch_size,
)
```

### 2. 过滤损坏图像

真实场景中图像易出现编码损坏，需过滤掉头部不含"JFIF"标识的无效JPEG文件，避免训练报错：

```Python

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)
print(f"Deleted {num_skipped} images.")
```

### 3. 数据增强实现

小数据集训练易过拟合，通过**随机水平翻转**、**小角度旋转**引入样本多样性，本次采用**函数式**实现数据增强（可灵活作用于模型或数据集）：

```Python

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),  # 水平翻转
    layers.RandomRotation(0.1),       # 随机旋转10%角度
]
def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
```

### 4. 数据集性能优化

为避免训练时的I/O阻塞，利用`tensorflow.data`的**并行映射**和**预取**功能，让数据预处理和模型训练异步进行，最大化硬件利用率：

```Python

import tensorflow as tf
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,  # 自动适配并行数
)
# 预取数据到GPU内存，消除数据读取瓶颈
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
```

## 二、数据预处理的两种核心方式

Keras中图像标准化（归一化）和数据增强的结合有两种主流方式，各有适用场景，需根据训练硬件选择：

### 方式1：作为模型的一部分

将数据增强和Rescaling层嵌入模型，**利用GPU加速**，仅在训练时生效（测试/推理时自动关闭增强），适合GPU训练场景：

```Python

inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)  # 增强层
x = layers.Rescaling(1./255)(x) # 归一化到[0,1]
# 后续模型层...
```

### 方式2：作用于数据集（本次选用）

通过`map`将增强应用到数据集，**在CPU异步执行**，预处理后的批次数据直接传入模型，是CPU训练的优选，也是通用稳妥的方案：

```Python

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
```

**关键原则**：数据增强需在**归一化前**执行，避免增强操作破坏归一化后的数值范围。

## 三、模型架构设计与实现

本次构建**轻量版Xception网络**，核心采用**可分离卷积（SeparableConv2D）+残差连接**，兼顾模型性能和计算效率，适配猫狗分类的简单场景，同时加入BatchNormalization、Dropout缓解过拟合，网络为**函数式API**构建，灵活性更高。

### 1. 模型整体架构

模型输入为`(180,180,3)`的RGB图像，整体分为**输入层-入口块-残差块序列-全局池化-分类头**，各模块输出形状和核心操作如下：

```Plain Text

InputLayer          (None, 180, 180, 3)   # 输入：180*180*3 RGB图像
Rescaling           (None, 180, 180, 3)   # 归一化：1/255
Conv2D+BN+Activation(None, 90, 90, 128)   # 入口块：下采样到90*90
├─残差块1（256）     (None, 45, 45, 256)   # 可分离卷积+池化+残差连接
├─残差块2（512）     (None, 23, 23, 512)
├─残差块3（728）     (None, 12, 12, 728)
├─可分离卷积（1024） (None, 12, 12, 1024)
├─GlobalAveragePooling2D (None, 1024)    # 全局平均池化：展平为特征向量
├─Dropout(0.25)      (None, 1024)         # 随机失活：缓解过拟合
└─Dense(1)           (None, 1)            # 二分类输出头：logits形式
```

### 2. 核心架构特点

1. **可分离卷积**：将标准卷积拆分为深度卷积+点卷积，减少参数量和计算量，不损失特征提取能力；

2. **残差连接**：解决深层网络梯度消失问题，通过`layers.add`将前一块的特征投影后与当前块特征融合；

3. **批归一化（BatchNormalization）**：加速训练收敛，稳定分布，减少初始化依赖；

4. **全局平均池化**：替代全连接层展平，减少参数量，避免过拟合；

5. **输出为Logits**：分类头不设置激活函数，训练时结合`from_logits=True`的损失函数，提升数值稳定性。

### 3. 模型构建代码

通过自定义函数封装模型，支持灵活修改输入形状和类别数，适配不同分类任务：

```Python

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # 入口块：归一化+卷积+批归一化+激活
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # 保存残差连接的基准特征
    # 残差块序列
    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        # 下采样
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # 残差投影：匹配特征维度和尺寸
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # 残差融合
        previous_block_activation = x  # 更新基准特征

    # 尾部卷积+池化
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    # 分类头
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes if num_classes>2 else 1, activation=None)(x)
    return keras.Model(inputs, outputs)

# 实例化模型：input_shape为(image_size + 通道数)
model = make_model(input_shape=image_size + (3,), num_classes=2)
```

## 四、模型训练与配置

### 1. 训练参数配置

- 优化器：Adam，学习率设为3e-4（小学习率适合轻量模型，避免训练震荡）；

- 损失函数：BinaryCrossentropy（二分类），`from_logits=True`匹配模型logits输出；

- 评价指标：BinaryAccuracy（二分类准确率）；

- 回调函数：ModelCheckpoint，保存各epoch的模型，便于后续选优；

- 训练轮数：25epoch，猫狗数据集该轮数可达到90%+验证准确率。

### 2. 训练代码

```Python

epochs = 25
# 回调函数：保存模型
callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]
# 模型编译
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
# 开始训练
history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
```

### 3. 训练结果

25轮训练后，训练集准确率约96.38%、损失0.0903，验证集准确率93.82%、损失0.1542，无明显过拟合，说明数据增强和Dropout的正则化效果有效；若需进一步提升性能，可增加训练轮数（50+）或微调模型超参数（如学习率、批次大小）。

## 五、模型推理（新数据预测）

推理阶段**数据增强和Dropout会自动失效**，只需将新图像做**尺寸统一-数组转换-批次扩展**，再传入模型预测，最后通过sigmoid将logits转换为0-1的概率值，判断猫狗类别。

### 推理代码

```Python

# 加载并预处理单张图像
img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # 增加批次维度：(1,180,180,3)

# 预测并转换为概率
predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))  # sigmoid转概率
# 输出结果
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```

## 六、核心知识点总结与心得

1. **数据集处理**：Keras的`image_dataset_from_directory`是处理图像分类任务的高效工具，可一键完成划分、尺寸统一，结合`tensorflow.data`的`map`和`prefetch`能大幅提升训练效率；

2. **数据增强**：小数据集的必备操作，需遵循**增强在前，归一化在后**的原则，两种实现方式可根据硬件灵活选择；

3. **模型设计**：轻量场景下可分离卷积替代标准卷积，残差连接提升深层网络性能，BatchNormalization和Dropout是常用的正则化手段，输出层用Logits形式能提升训练稳定性；

4. **训练技巧**：Adam优化器的小学习率（3e-4）适合计算机视觉轻量模型，ModelCheckpoint回调函数可保存训练过程中的模型，便于后续复现和选优；

5. **推理要点**：新图像需和训练数据做**相同的尺寸预处理**，必须扩展批次维度（模型默认接收批次输入），logits需通过sigmoid（二分类）/softmax（多分类）转换为概率。

本次实战从0到1实现了猫狗图像分类，掌握了Keras构建图像分类模型的完整流程，后续可将该框架迁移到其他二分类图像任务，也可通过增加模型深度、使用预训练权重进一步提升复杂场景的分类性能。
> （注：文档部分内容可能由 AI 生成）