import React from 'react';
import { motion } from 'framer-motion';
import { MatrixVisualization } from '../components/Matrix/MatrixVisualization';
import { TransformerSimulator } from '../utils/transformer';
import { addMatrices, multiplyMatrices, relu, createRandomMatrix, normalizeMatrix } from '../utils/matrix';

interface FFNProps {
  tokens: any[];
  simulator: TransformerSimulator;
}

export function FFN({ tokens, simulator }: FFNProps) {
  const embeddingMatrix = simulator.embeddingLookup(tokens);
  const positionalMatrix = simulator.positionalEncoding(tokens.length);
  const X = addMatrices(embeddingMatrix, positionalMatrix);
  const attentionOutput = simulator.multiHeadAttention(X);
  const normalized = normalizeMatrix(addMatrices(X, attentionOutput));
  
  const dModel = 512;
  const dFF = dModel * 4;
  
  const W1 = createRandomMatrix(dModel, dFF, 'weight');
  const W2 = createRandomMatrix(dFF, dModel, 'weight');
  
  const hidden = multiplyMatrices(normalized, W1);
  const activated = relu(hidden);
  const output = multiplyMatrices(activated, W2);

  return (
    <div className="flex flex-col items-center gap-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-monokai-fg font-mono mb-4">Feed-Forward Network</h2>
        <p className="text-monokai-comment font-mono text-sm max-w-md">
          前馈神经网络：对每个位置独立地进行非线性变换。
        </p>
      </div>

      <div className="flex flex-col items-center gap-4">
        <div className="flex items-center gap-4 flex-wrap justify-center">
          <MatrixVisualization matrix={normalized} title="Input [seq_len, 512]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
          <motion.div className="text-monokai-blue text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>×</motion.div>
          <MatrixVisualization matrix={W1} title="W1 [512, 2048]" maxDisplayRows={6} maxDisplayCols={6} />
          <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>=</motion.div>
          <MatrixVisualization matrix={hidden} title="Hidden [seq_len, 2048]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
        </div>

        <motion.div 
          className="text-monokai-orange text-4xl font-bold"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 0.5, repeat: Infinity }}
        >
          ReLU
        </motion.div>

        <div className="flex items-center gap-4 flex-wrap justify-center">
          <MatrixVisualization matrix={activated} title="After ReLU [seq_len, 2048]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
          <motion.div className="text-monokai-blue text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>×</motion.div>
          <MatrixVisualization matrix={W2} title="W2 [2048, 512]" maxDisplayRows={6} maxDisplayCols={6} />
          <motion.div className="text-monokai-green text-3xl" animate={{ x: [0, 10, 0] }} transition={{ duration: 1, repeat: Infinity }}>=</motion.div>
          <MatrixVisualization matrix={output} title="Output [seq_len, 512]" maxDisplayRows={tokens.length} maxDisplayCols={6} />
        </div>
      </div>

      <div className="flex gap-4 flex-wrap justify-center">
        <div className="p-4 bg-monokai-bg border-2 border-monokai-orange rounded-lg max-w-sm">
          <h3 className="text-lg font-semibold text-monokai-orange font-mono mb-2">ReLU 激活函数</h3>
          <p className="text-sm text-monokai-comment font-mono leading-relaxed">
            ReLU(x) = max(0, x)
            将所有负数截断为 0，保留正数不变。
            这引入了非线性，使模型能够学习复杂的模式。
          </p>
        </div>
        <div className="p-4 bg-monokai-bg border-2 border-monokai-blue rounded-lg max-w-sm">
          <h3 className="text-lg font-semibold text-monokai-blue font-mono mb-2">扩展与压缩</h3>
          <p className="text-sm text-monokai-comment font-mono leading-relaxed">
            先将维度从 512 扩展到 2048（4倍），再压缩回 512。
            这种"扩展-压缩"结构增加了模型容量，
            使其能够学习更丰富的特征表示。
          </p>
        </div>
      </div>

      <div className="p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg max-w-lg">
        <h3 className="text-lg font-semibold text-monokai-fg font-mono mb-2">完整流程</h3>
        <p className="text-sm text-monokai-comment font-mono leading-relaxed">
          FFN(x) = ReLU(xW1 + b1)W2 + b2
          <br /><br />
          每个位置独立地通过相同的 FFN，不与其他位置交互。
          这与自注意力不同，自注意力允许所有位置之间交互。
        </p>
      </div>
    </div>
  );
}