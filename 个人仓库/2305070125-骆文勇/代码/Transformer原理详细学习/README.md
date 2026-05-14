# Transformer 原理交互式可视化

这是一个交互式的 Transformer 模型可视化演示网站，用于直观地展示 Transformer 模型在处理文本时的内部矩阵运算过程。

## 功能特点

- **完整的 Transformer 流程可视化**：从 Tokenization 到 FFN 的完整流程
- **自注意力机制详细拆解**：6 个子步骤逐步展示自注意力机制
- **交互式矩阵展示**：可悬停查看详细信息
- **热力图可视化**：直观展示注意力权重
- **Monokai 配色方案**：程序员熟悉的深色主题
- **实时解说**：每个步骤都有详细的中文解说

## 技术栈

- React 18 + TypeScript
- Vite
- Tailwind CSS
- Framer Motion (动画)
- D3.js (数据可视化)

## 安装与运行

1. 安装依赖：
```bash
npm install
```

2. 启动开发服务器：
```bash
npm run dev
```

3. 打开浏览器访问 `http://localhost:3000`

## 使用说明

1. **左侧输入区**：输入你想要分析的英文句子
2. **中间可视化舞台**：查看当前的矩阵运算过程
3. **右侧控制面板**：
   - 播放/暂停动画
   - 调整进度
   - 切换处理阶段
   - 查看实时解说

## 可视化阶段

1. **Tokenization**：文本分词
2. **Embedding**：词向量查找
3. **Positional Encoding**：位置编码
4. **Self-Attention**：自注意力机制
   - 线性投影 (Q, K, V 生成)
   - 注意力分数计算
   - 缩放与 Mask
   - Softmax
   - 加权求和
   - 多头注意力
5. **Add & Norm**：残差连接与层归一化
6. **FFN**：前馈神经网络

## 项目结构

```
src/
├── components/          # 可复用组件
│   ├── InputPanel/     # 输入面板
│   ├── VisualizationStage/  # 可视化舞台
│   ├── ControlPanel/   # 控制面板
│   ├── Matrix/         # 矩阵可视化组件
│   └── Heatmap/        # 热力图组件
├── stages/             # 各个可视化阶段
│   ├── Tokenization.tsx
│   ├── Embedding.tsx
│   ├── PositionalEncoding.tsx
│   ├── SelfAttention.tsx
│   ├── AddNorm.tsx
│   └── FFN.tsx
├── utils/              # 工具函数
│   ├── matrix.ts       # 矩阵运算
│   ├── transformer.ts  # Transformer 模拟
│   └── colors.ts       # 配色方案
├── context/            # React Context
│   └── VisualizationContext.tsx
├── types/              # TypeScript 类型定义
│   └── index.ts
├── App.tsx
└── main.tsx
```

## 构建生产版本

```bash
npm run build
```

构建后的文件将输出到 `dist/` 目录。