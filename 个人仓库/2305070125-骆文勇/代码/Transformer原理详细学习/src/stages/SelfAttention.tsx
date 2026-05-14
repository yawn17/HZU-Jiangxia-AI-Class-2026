import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { MatrixVisualization } from '../components/Matrix/MatrixVisualization';
import { Heatmap } from '../components/Heatmap/Heatmap';
import { TransformerSimulator } from '../utils/transformer';
import { addMatrices } from '../utils/matrix';
import { AttentionStep } from '../types';

interface SelfAttentionProps {
  tokens: any[];
  simulator: TransformerSimulator;
  attentionStep: AttentionStep;
}

export function SelfAttention({ tokens, simulator, attentionStep }: SelfAttentionProps) {
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | undefined>();
  
  const embeddingMatrix = simulator.embeddingLookup(tokens);
  const positionalMatrix = simulator.positionalEncoding(tokens.length);
  const X = addMatrices(embeddingMatrix, positionalMatrix);
  
  const { Q, K, V } = simulator.linearProjection(X);
  const attentionScores = simulator.computeAttentionScores(Q, K);
  const scaledScores = simulator.scaleAttentionScores(attentionScores);
  const attentionWeights = simulator.applySoftmax(scaledScores);
  const weightedSum = simulator.weightedSum(attentionWeights, V);
  const multiHeadOutput = simulator.multiHeadAttention(X);

  const stepDescriptions: Record<AttentionStep, string> = {
    'linear-projection': '通过三个权重矩阵 Wq, Wk, Wv 将输入向量投影到 Query、Key、Value 空间。',
    'attention-scores': '计算 Query 和 Key 的点积，得到注意力分数矩阵。',
    'scaling-masking': '将注意力分数除以 √d_k 进行缩放。',
    'softmax': '对每行的注意力分数应用 Softmax 函数。',
    'weighted-sum': '将注意力权重与 Value 向量相乘。',
    'multi-head': '多头注意力：重复上述过程 8 次，然后拼接输出。'
  };

  const renderLinearProjection = () => (
    <div className="flex flex-col items-center gap-6">
      <div className="flex items-center gap-6 flex-wrap justify-center">
        <MatrixVisualization matrix={X} title="X [seq_len, 512]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
        <motion.div className="text-monokai-blue text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>×</motion.div>
        <MatrixVisualization matrix={simulator.getWQ()} title="Wq [512, 64]" maxDisplayRows={6} maxDisplayCols={6} />
        <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>=</motion.div>
        <MatrixVisualization matrix={Q} title="Q [seq_len, 64]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
      </div>
      <div className="flex items-center gap-6 flex-wrap justify-center">
        <MatrixVisualization matrix={X} title="X [seq_len, 512]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
        <motion.div className="text-monokai-blue text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>×</motion.div>
        <MatrixVisualization matrix={simulator.getWK()} title="Wk [512, 64]" maxDisplayRows={6} maxDisplayCols={6} />
        <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>=</motion.div>
        <MatrixVisualization matrix={K} title="K [seq_len, 64]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
      </div>
      <div className="flex items-center gap-6 flex-wrap justify-center">
        <MatrixVisualization matrix={X} title="X [seq_len, 512]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
        <motion.div className="text-monokai-blue text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>×</motion.div>
        <MatrixVisualization matrix={simulator.getWV()} title="Wv [512, 64]" maxDisplayRows={6} maxDisplayCols={6} />
        <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>=</motion.div>
        <MatrixVisualization matrix={V} title="V [seq_len, 64]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
      </div>
    </div>
  );

  const renderAttentionScores = () => (
    <div className="flex flex-col items-center gap-6">
      <div className="flex items-center gap-6 flex-wrap justify-center">
        <MatrixVisualization matrix={Q} title="Q [seq_len, 64]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
        <motion.div className="text-monokai-blue text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>×</motion.div>
        <MatrixVisualization matrix={K} title="K^T [64, seq_len]" maxDisplayRows={6} maxDisplayCols={tokens.length} />
        <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>=</motion.div>
        <Heatmap 
          matrix={attentionScores} 
          title="Attention Scores [seq_len, seq_len]"
          labels={{ 
            rows: tokens.map(t => t.text), 
            cols: tokens.map(t => t.text) 
          }}
          highlightedCell={selectedCell}
          onCellHover={(row, col) => setSelectedCell({ row, col })}
          onCellLeave={() => setSelectedCell(undefined)}
        />
      </div>
      {selectedCell && (
        <div className="p-4 bg-monokai-bg border-2 border-monokai-blue rounded-lg">
          <p className="text-monokai-fg font-mono text-sm">
            "{tokens[selectedCell.row].text}" 的 Query 与 "{tokens[selectedCell.col].text}" 的 Key 的点积
          </p>
          <p className="text-monokai-blue font-mono text-lg mt-2">
            Score: {attentionScores.data[selectedCell.row][selectedCell.col].toFixed(3)}
          </p>
        </div>
      )}
    </div>
  );

  const renderScalingMasking = () => (
    <div className="flex flex-col items-center gap-6">
      <div className="flex items-center gap-6 flex-wrap justify-center">
        <Heatmap 
          matrix={attentionScores} 
          title="Before Scaling"
          labels={{ 
            rows: tokens.map(t => t.text), 
            cols: tokens.map(t => t.text) 
          }}
        />
        <motion.div className="text-monokai-blue text-3xl" animate={{ scale: [1, 1.2, 1] }} transition={{ duration: 1, repeat: Infinity }}>÷ √64</motion.div>
        <Heatmap 
          matrix={scaledScores} 
          title="After Scaling"
          labels={{ 
            rows: tokens.map(t => t.text), 
            cols: tokens.map(t => t.text) 
          }}
        />
      </div>
      <div className="p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg max-w-lg">
        <p className="text-sm text-monokai-comment font-mono leading-relaxed">
          缩放因子 √d_k = √64 = 8。缩放后的数值变小，防止点积过大导致梯度消失。
          注意观察热力图颜色的变化。
        </p>
      </div>
    </div>
  );

  const renderSoftmax = () => (
    <div className="flex flex-col items-center gap-6">
      <div className="flex items-center gap-6 flex-wrap justify-center">
        <Heatmap 
          matrix={scaledScores} 
          title="Before Softmax"
          labels={{ 
            rows: tokens.map(t => t.text), 
            cols: tokens.map(t => t.text) 
          }}
        />
        <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>→</motion.div>
        <Heatmap 
          matrix={attentionWeights} 
          title="After Softmax"
          labels={{ 
            rows: tokens.map(t => t.text), 
            cols: tokens.map(t => t.text) 
          }}
        />
      </div>
      <div className="p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg max-w-lg">
        <p className="text-sm text-monokai-comment font-mono leading-relaxed">
          Softmax 将每行的数值转换为概率分布（0-1 之间，每行和为 1）。
          只有少数几个格子会变得很亮（权重高），表示这个词主要关注哪些其他词。
        </p>
      </div>
    </div>
  );

  const renderWeightedSum = () => (
    <div className="flex flex-col items-center gap-6">
      <div className="flex items-center gap-6 flex-wrap justify-center">
        <Heatmap 
          matrix={attentionWeights} 
          title="Attention Weights [seq_len, seq_len]"
          labels={{ 
            rows: tokens.map(t => t.text), 
            cols: tokens.map(t => t.text) 
          }}
        />
        <motion.div className="text-monokai-blue text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>×</motion.div>
        <MatrixVisualization matrix={V} title="V [seq_len, 64]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
        <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>=</motion.div>
        <MatrixVisualization matrix={weightedSum} title="Output [seq_len, 64]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
      </div>
      <div className="p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg max-w-lg">
        <p className="text-sm text-monokai-comment font-mono leading-relaxed">
          加权求和：每个词的输出是所有词的 Value 按照注意力权重的加权和。
          例如，"cat" 的输出可能主要由 "cat" 和 "sat" 的 Value 组成。
        </p>
      </div>
    </div>
  );

  const renderMultiHead = () => (
    <div className="flex flex-col items-center gap-6">
      <div className="text-center">
        <h3 className="text-lg font-semibold text-monokai-fg font-mono mb-2">8 个注意力头并行处理</h3>
        <div className="flex gap-2 flex-wrap justify-center">
          {Array.from({ length: 8 }).map((_, i) => (
            <motion.div
              key={i}
              className="w-16 h-16 rounded-lg flex items-center justify-center font-mono font-bold"
              style={{ 
                backgroundColor: `hsl(${i * 45}, 70%, 50%)`,
                color: '#272822'
              }}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: i * 0.1 }}
            >
              Head {i + 1}
            </motion.div>
          ))}
        </div>
      </div>
      <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>↓ Concat ↓</motion.div>
      <MatrixVisualization matrix={multiHeadOutput} title="Multi-Head Output [seq_len, 512]" maxDisplayRows={tokens.length} maxDisplayCols={8} />
      <div className="p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg max-w-lg">
        <p className="text-sm text-monokai-comment font-mono leading-relaxed">
          多头注意力：将单头注意力过程重复 8 次，每次使用不同的投影矩阵。
          然后将 8 个头的输出在最后一个维度拼接，再通过一个线性层得到最终输出。
          这使得模型能够同时关注不同类型的关系。
        </p>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-monokai-fg font-mono mb-2">Self-Attention</h2>
        <p className="text-monokai-comment font-mono text-sm max-w-md">
          自注意力机制：让每个词都能"看到"句子中的其他词，捕捉词与词之间的关系。
        </p>
      </div>

      <div className="p-3 bg-monokai-orange/20 border-2 border-monokai-orange rounded-lg">
        <p className="text-monokai-orange font-mono text-sm text-center">
          {stepDescriptions[attentionStep]}
        </p>
      </div>

      {attentionStep === 'linear-projection' && renderLinearProjection()}
      {attentionStep === 'attention-scores' && renderAttentionScores()}
      {attentionStep === 'scaling-masking' && renderScalingMasking()}
      {attentionStep === 'softmax' && renderSoftmax()}
      {attentionStep === 'weighted-sum' && renderWeightedSum()}
      {attentionStep === 'multi-head' && renderMultiHead()}
    </div>
  );
}