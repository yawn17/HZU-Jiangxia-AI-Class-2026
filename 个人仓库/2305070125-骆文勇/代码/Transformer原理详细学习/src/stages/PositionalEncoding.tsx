import React from 'react';
import { motion } from 'framer-motion';
import { MatrixVisualization } from '../components/Matrix/MatrixVisualization';
import { TransformerSimulator } from '../utils/transformer';
import { addMatrices } from '../utils/matrix';

interface PositionalEncodingProps {
  tokens: any[];
  simulator: TransformerSimulator;
}

export function PositionalEncoding({ tokens, simulator }: PositionalEncodingProps) {
  const embeddingMatrix = simulator.embeddingLookup(tokens);
  const positionalMatrix = simulator.positionalEncoding(tokens.length);
  const combinedMatrix = addMatrices(embeddingMatrix, positionalMatrix);

  return (
    <div className="flex flex-col items-center gap-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-monokai-fg font-mono mb-4">Positional Encoding</h2>
        <p className="text-monokai-comment font-mono text-sm max-w-md">
          添加位置编码，让模型知道词在句子中的位置信息。
        </p>
      </div>

      <div className="flex items-center gap-4 flex-wrap justify-center">
        <div className="flex flex-col items-center gap-2">
          <h3 className="text-lg font-semibold text-monokai-fg font-mono">Embedding</h3>
          <MatrixVisualization 
            matrix={embeddingMatrix}
            maxDisplayRows={tokens.length}
            maxDisplayCols={6}
          />
        </div>

        <motion.div 
          className="text-monokai-blue text-4xl"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          +
        </motion.div>

        <div className="flex flex-col items-center gap-2">
          <h3 className="text-lg font-semibold text-monokai-fg font-mono">Positional</h3>
          <MatrixVisualization 
            matrix={positionalMatrix}
            maxDisplayRows={tokens.length}
            maxDisplayCols={6}
          />
        </div>

        <motion.div 
          className="text-monokai-green text-4xl"
          animate={{ x: [0, 10, 0] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          =
        </motion.div>

        <div className="flex flex-col items-center gap-2">
          <h3 className="text-lg font-semibold text-monokai-fg font-mono">Combined</h3>
          <MatrixVisualization 
            matrix={combinedMatrix}
            maxDisplayRows={tokens.length}
            maxDisplayCols={6}
          />
        </div>
      </div>

      <div className="mt-8 p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg max-w-lg">
        <h3 className="text-lg font-semibold text-monokai-fg font-mono mb-2">说明</h3>
        <p className="text-sm text-monokai-comment font-mono leading-relaxed">
          Transformer 没有循环结构，无法像 RNN 那样自然地知道词的位置信息。
          位置编码使用正弦和余弦函数为每个位置生成独特的模式。
          不同频率的波形组合，使得每个位置都有唯一的编码。
        </p>
      </div>
    </div>
  );
}