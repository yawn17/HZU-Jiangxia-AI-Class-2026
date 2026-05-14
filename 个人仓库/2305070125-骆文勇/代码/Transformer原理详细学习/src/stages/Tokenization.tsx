import React from 'react';
import { motion } from 'framer-motion';

interface TokenizationProps {
  tokens: any[];
}

export function Tokenization({ tokens }: TokenizationProps) {
  return (
    <div className="flex flex-col items-center gap-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-monokai-fg font-mono mb-4">Tokenization</h2>
        <p className="text-monokai-comment font-mono text-sm max-w-md">
          将文本拆分为单词或子词，每个词被分配一个唯一的整数 ID。
        </p>
      </div>

      <div className="flex items-center gap-4">
        <div className="text-monokai-fg font-mono text-lg">
          "{tokens.map(t => t.text).join(' ')}"
        </div>
        <motion.div 
          className="text-monokai-blue text-4xl"
          animate={{ x: [0, 10, 0] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          →
        </motion.div>
      </div>

      <div className="flex flex-wrap gap-4 justify-center">
        {tokens.map((token, index) => (
          <motion.div
            key={index}
            className="flex flex-col items-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.2 }}
          >
            <div className="px-6 py-4 bg-monokai-orange/20 border-2 border-monokai-orange rounded-lg mb-2">
              <div className="text-monokai-fg font-mono text-xl font-bold">{token.text}</div>
            </div>
            <div className="px-4 py-2 bg-monokai-blue/20 border-2 border-monokai-blue rounded-lg">
              <div className="text-monokai-blue font-mono text-sm">ID: {token.id}</div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-8 p-4 bg-monokai-bg border-2 border-monokai-comment rounded-lg max-w-lg">
        <h3 className="text-lg font-semibold text-monokai-fg font-mono mb-2">说明</h3>
        <p className="text-sm text-monokai-comment font-mono leading-relaxed">
          Tokenization 是 NLP 的第一步。计算机无法直接理解文本，所以需要将文本转换为数字。
          每个词被分配一个唯一的 ID，这个 ID 将用于在词表中查找对应的向量表示。
        </p>
      </div>
    </div>
  );
}