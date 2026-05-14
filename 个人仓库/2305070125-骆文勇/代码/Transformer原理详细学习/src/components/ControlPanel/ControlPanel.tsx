import React from 'react';
import { motion } from 'framer-motion';
import { Stage, AttentionStep } from '../../types';
import { useVisualization } from '../../context/VisualizationContext';

const stageLabels: Record<Stage, string> = {
  'tokenization': 'Tokenization',
  'embedding': 'Embedding',
  'positional-encoding': 'Positional Encoding',
  'self-attention': 'Self-Attention',
  'add-norm': 'Add & Norm',
  'ffn': 'FFN'
};

const attentionStepLabels: Record<AttentionStep, string> = {
  'linear-projection': '线性投影 (Q, K, V)',
  'attention-scores': '注意力分数',
  'scaling-masking': '缩放与 Mask',
  'softmax': 'Softmax',
  'weighted-sum': '加权求和',
  'multi-head': '多头注意力'
};

const stageDescriptions: Record<Stage, string> = {
  'tokenization': '将文本拆分为单词或子词，每个词被分配一个唯一的整数 ID。',
  'embedding': '从词表中查找每个词对应的向量表示，将离散的 ID 映射到连续的向量空间。',
  'positional-encoding': '添加位置编码，让模型知道词在句子中的位置信息。',
  'self-attention': '自注意力机制：让每个词都能"看到"句子中的其他词，捕捉词与词之间的关系。',
  'add-norm': '残差连接和层归一化：帮助梯度流动，稳定训练过程。',
  'ffn': '前馈神经网络：对每个位置独立地进行非线性变换。'
};

const attentionStepDescriptions: Record<AttentionStep, string> = {
  'linear-projection': '通过三个权重矩阵 Wq, Wk, Wv 将输入向量投影到 Query、Key、Value 空间。Query 表示"我在找什么"，Key 表示"我包含什么信息"，Value 表示"我有什么内容"。',
  'attention-scores': '计算 Query 和 Key 的点积，得到注意力分数矩阵。分数越高表示两个词之间的关联越强。',
  'scaling-masking': '将注意力分数除以 √d_k 进行缩放，防止点积过大导致梯度消失。Decoder 中还会应用 Mask 防止看到未来信息。',
  'softmax': '对每行的注意力分数应用 Softmax 函数，将其转换为概率分布（0-1 之间，每行和为 1）。只有少数几个格子会变得很亮（权重高）。',
  'weighted-sum': '将 Softmax 后的注意力权重与 Value 向量相乘，得到每个词的输出。输出是所有词的 Value 按照权重的加权和。',
  'multi-head': '多头注意力：将上述过程重复 8 次，每次使用不同的投影矩阵。然后将 8 个头的输出拼接起来，再通过一个线性层得到最终输出。'
};

export function ControlPanel() {
  const { state, nextStage, prevStage, togglePlay, reset, updateProgress } = useVisualization();

  const stages: Stage[] = ['tokenization', 'embedding', 'positional-encoding', 'self-attention', 'add-norm', 'ffn'];
  const attentionSteps: AttentionStep[] = ['linear-projection', 'attention-scores', 'scaling-masking', 'softmax', 'weighted-sum', 'multi-head'];

  const handleStageClick = (stage: Stage) => {
    if (stage === state.currentStage) return;
    
    const currentIndex = stages.indexOf(state.currentStage);
    const targetIndex = stages.indexOf(stage);
    
    if (targetIndex > currentIndex) {
      for (let i = 0; i < targetIndex - currentIndex; i++) {
        nextStage();
      }
    } else {
      for (let i = 0; i < currentIndex - targetIndex; i++) {
        prevStage();
      }
    }
  };

  const handleAttentionStepClick = (step: AttentionStep) => {
    if (step === state.attentionStep) return;
    
    const currentIndex = attentionSteps.indexOf(state.attentionStep);
    const targetIndex = attentionSteps.indexOf(step);
    
    if (targetIndex > currentIndex) {
      for (let i = 0; i < targetIndex - currentIndex; i++) {
        nextStage();
      }
    } else {
      for (let i = 0; i < currentIndex - targetIndex; i++) {
        prevStage();
      }
    }
  };

  return (
    <motion.div 
      className="w-80 bg-monokai-bg border-l-2 border-monokai-comment p-6 flex flex-col gap-6"
      initial={{ x: 300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div>
        <h2 className="text-xl font-bold text-monokai-fg mb-4 font-mono">控制面板</h2>
        
        <div className="flex gap-2 mb-6">
          <button
            onClick={togglePlay}
            className="flex-1 px-4 py-2 bg-monokai-blue/20 border-2 border-monokai-blue rounded-lg text-monokai-blue font-mono hover:bg-monokai-blue/30 transition-colors"
          >
            {state.isPlaying ? '⏸ 暂停' : '▶ 播放'}
          </button>
          <button
            onClick={reset}
            className="px-4 py-2 bg-monokai-red/20 border-2 border-monokai-red rounded-lg text-monokai-red font-mono hover:bg-monokai-red/30 transition-colors"
          >
            ↺ 重置
          </button>
        </div>

        <div className="mb-4">
          <label className="text-sm text-monokai-comment font-mono mb-2 block">进度</label>
          <input
            type="range"
            min="0"
            max="100"
            value={state.progress}
            onChange={(e) => updateProgress(Number(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-monokai-comment font-mono mt-1">{state.progress}%</div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={prevStage}
            className="flex-1 px-3 py-2 bg-monokai-comment/20 border-2 border-monokai-comment rounded-lg text-monokai-fg font-mono hover:bg-monokai-comment/30 transition-colors"
          >
            ← 上一步
          </button>
          <button
            onClick={nextStage}
            className="flex-1 px-3 py-2 bg-monokai-green/20 border-2 border-monokai-green rounded-lg text-monokai-green font-mono hover:bg-monokai-green/30 transition-colors"
          >
            下一步 →
          </button>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-monokai-fg mb-3 font-mono">处理阶段</h3>
        <div className="space-y-2">
          {stages.map((stage) => (
            <motion.button
              key={stage}
              onClick={() => handleStageClick(stage)}
              className={`w-full px-3 py-2 rounded-lg text-left font-mono text-sm transition-all ${
                state.currentStage === stage
                  ? 'bg-monokai-purple/20 border-2 border-monokai-purple text-monokai-purple'
                  : 'bg-monokai-bg border-2 border-monokai-comment text-monokai-fg hover:border-monokai-purple/50'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {stageLabels[stage]}
            </motion.button>
          ))}
        </div>
      </div>

      {state.currentStage === 'self-attention' && (
        <div>
          <h3 className="text-lg font-semibold text-monokai-fg mb-3 font-mono">自注意力步骤</h3>
          <div className="space-y-2">
            {attentionSteps.map((step) => (
              <motion.button
                key={step}
                onClick={() => handleAttentionStepClick(step)}
                className={`w-full px-3 py-2 rounded-lg text-left font-mono text-xs transition-all ${
                  state.attentionStep === step
                    ? 'bg-monokai-orange/20 border-2 border-monokai-orange text-monokai-orange'
                    : 'bg-monokai-bg border-2 border-monokai-comment text-monokai-fg hover:border-monokai-orange/50'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {attentionStepLabels[step]}
              </motion.button>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1">
        <h3 className="text-lg font-semibold text-monokai-fg mb-3 font-mono">解说</h3>
        <div className="p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg">
          <p className="text-sm text-monokai-fg leading-relaxed">
            {state.currentStage === 'self-attention' 
              ? attentionStepDescriptions[state.attentionStep]
              : stageDescriptions[state.currentStage]
            }
          </p>
        </div>
      </div>
    </motion.div>
  );
}