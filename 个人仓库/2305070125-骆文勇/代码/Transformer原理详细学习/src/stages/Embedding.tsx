import React from 'react';
import { motion } from 'framer-motion';
import { MatrixVisualization } from '../components/Matrix/MatrixVisualization';
import { TransformerSimulator } from '../utils/transformer';

interface EmbeddingProps {
  tokens: any[];
  simulator: TransformerSimulator;
}

export function Embedding({ tokens, simulator }: EmbeddingProps) {
  const embeddingMatrix = simulator.embeddingLookup(tokens);

  return (
    <div className="flex flex-col items-center gap-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-monokai-fg font-mono mb-4">Embedding Lookup</h2>
        <p className="text-monokai-comment font-mono text-sm max-w-md">
          从词表中查找每个词对应的向量表示，将离散的 ID 映射到连续的向量空间。
        </p>
      </div>

      <div className="flex items-center gap-8">
        <div className="flex flex-col items-center gap-4">
          <h3 className="text-lg font-semibold text-monokai-fg font-mono">Token IDs</h3>
          <div className="flex flex-col gap-2">
            {tokens.map((token, index) => (
              <motion.div
                key={index}
                className="px-4 py-2 bg-monokai-blue/20 border-2 border-monokai-blue rounded-lg"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.2 }}
              >
                <div className="text-monokai-fg font-mono text-sm">{token.text}</div>
                <div className="text-monokai-blue font-mono text-xs">ID: {token.id}</div>
              </motion.div>
            ))}
          </div>
        </div>

        <motion.div 
          className="text-monokai-green text-4xl"
          animate={{ x: [0, 10, 0] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          →
        </motion.div>

        <div className="flex flex-col items-center gap-4">
          <h3 className="text-lg font-semibold text-monokai-fg font-mono">Embedding Matrix</h3>
          <MatrixVisualization 
            matrix={embeddingMatrix}
            title="X [seq_len, d_model]"
            maxDisplayRows={tokens.length}
            maxDisplayCols={8}
          />
        </div>
      </div>

      <div className="mt-8 p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg max-w-lg">
        <h3 className="text-lg font-semibold text-monokai-fg font-mono mb-2">说明</h3>
        <p className="text-sm text-monokai-comment font-mono leading-relaxed">
          Embedding 是一个巨大的查找表，每一行对应一个词的向量表示。
          向量中的每个数值代表该词在不同语义维度上的特征。
          例如，"cat" 和 "dog" 的向量在"动物"维度上会有相似的数值。
        </p>
      </div>
    </div>
  );
}