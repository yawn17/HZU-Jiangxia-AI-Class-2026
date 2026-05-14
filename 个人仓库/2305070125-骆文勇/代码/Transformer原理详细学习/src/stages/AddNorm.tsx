import React from 'react';
import { motion } from 'framer-motion';
import { MatrixVisualization } from '../components/Matrix/MatrixVisualization';
import { TransformerSimulator } from '../utils/transformer';
import { addMatrices, normalizeMatrix } from '../utils/matrix';

interface AddNormProps {
  tokens: any[];
  simulator: TransformerSimulator;
}

export function AddNorm({ tokens, simulator }: AddNormProps) {
  const embeddingMatrix = simulator.embeddingLookup(tokens);
  const positionalMatrix = simulator.positionalEncoding(tokens.length);
  const X = addMatrices(embeddingMatrix, positionalMatrix);
  const attentionOutput = simulator.multiHeadAttention(X);
  const added = addMatrices(X, attentionOutput);
  const normalized = normalizeMatrix(added);

  return (
    <div className="flex flex-col items-center gap-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-monokai-fg font-mono mb-4">Add & Norm</h2>
        <p className="text-monokai-comment font-mono text-sm max-w-md">
          残差连接和层归一化：帮助梯度流动，稳定训练过程。
        </p>
      </div>

      <div className="flex flex-col items-center gap-6">
        <div className="flex items-center gap-4 flex-wrap justify-center">
          <MatrixVisualization matrix={X} title="Input X" maxDisplayRows={tokens.length} maxDisplayCols={6} />
          <motion.div 
            className="text-monokai-blue text-4xl"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          >
            +
          </motion.div>
          <MatrixVisualization matrix={attentionOutput} title="Attention Output" maxDisplayRows={tokens.length} maxDisplayCols={6} />
          <motion.div 
            className="text-monokai-green text-4xl"
            animate={{ x: [0, 10, 0] }}
            transition={{ duration: 1, repeat: Infinity }}
          >
            =
          </motion.div>
          <MatrixVisualization matrix={added} title="X + Attention" maxDisplayRows={tokens.length} maxDisplayCols={6} />
        </div>

        <motion.div 
          className="text-monokai-purple text-4xl"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          ↓ Layer Norm ↓
        </motion.div>

        <MatrixVisualization matrix={normalized} title="Normalized Output" maxDisplayRows={tokens.length} maxDisplayCols={6} />
      </div>

      <div className="flex gap-4 flex-wrap justify-center">
        <div className="p-4 bg-monokai-bg border-2 border-monokai-blue rounded-lg max-w-sm">
          <h3 className="text-lg font-semibold text-monokai-blue font-mono mb-2">残差连接 (Add)</h3>
          <p className="text-sm text-monokai-comment font-mono leading-relaxed">
            将输入直接加到输出上：Output = X + Attention(X)。
            这就像两条河流汇聚，保留了原始信息，防止信息丢失。
            有助于梯度在深层网络中流动。
          </p>
        </div>
        <div className="p-4 bg-monokai-bg border-2 border-monokai-purple rounded-lg max-w-sm">
          <h3 className="text-lg font-semibold text-monokai-purple font-mono mb-2">层归一化 (Norm)</h3>
          <p className="text-sm text-monokai-comment font-mono leading-relaxed">
            对每个样本的所有特征进行归一化，使其均值为 0，方差为 1。
            这可以稳定训练过程，加速收敛。
            与 Batch Norm 不同，Layer Norm 不依赖于 batch size。
          </p>
        </div>
      </div>
    </div>
  );
}