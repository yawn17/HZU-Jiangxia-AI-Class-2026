import React from 'react';
import { motion } from 'framer-motion';
import { useVisualization } from '../../context/VisualizationContext';

export function InputPanel() {
  const { state, setState } = useVisualization();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newText = e.target.value;
    setState(prev => ({ ...prev, inputText: newText }));
  };

  return (
    <motion.div 
      className="w-80 bg-monokai-bg border-r-2 border-monokai-comment p-6 flex flex-col gap-6"
      initial={{ x: -300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div>
        <h2 className="text-xl font-bold text-monokai-fg mb-4 font-mono">输入文本</h2>
        <input
          type="text"
          value={state.inputText}
          onChange={handleInputChange}
          className="w-full p-3 bg-monokai-bg border-2 border-monokai-comment rounded-lg text-monokai-fg font-mono focus:outline-none focus:border-monokai-blue"
          placeholder="输入英文句子..."
        />
      </div>

      <div>
        <h3 className="text-lg font-semibold text-monokai-fg mb-3 font-mono">Tokenization</h3>
        <div className="flex flex-wrap gap-2">
          {state.inputText.split(/\s+/).map((word, index) => (
            <motion.div
              key={index}
              className="px-3 py-2 bg-monokai-orange/20 border-2 border-monokai-orange rounded-lg"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="text-monokai-fg font-mono text-sm">{word}</div>
              <div className="text-monokai-comment font-mono text-xs mt-1">ID: {Math.floor(Math.random() * 10000)}</div>
            </motion.div>
          ))}
        </div>
      </div>

      <div className="mt-auto">
        <h3 className="text-lg font-semibold text-monokai-fg mb-3 font-mono">模型参数</h3>
        <div className="space-y-2 text-sm font-mono text-monokai-comment">
          <div className="flex justify-between">
            <span>d_model:</span>
            <span className="text-monokai-blue">512</span>
          </div>
          <div className="flex justify-between">
            <span>d_k:</span>
            <span className="text-monokai-blue">64</span>
          </div>
          <div className="flex justify-between">
            <span>num_heads:</span>
            <span className="text-monokai-blue">8</span>
          </div>
          <div className="flex justify-between">
            <span>seq_len:</span>
            <span className="text-monokai-blue">{state.inputText.split(/\s+/).length}</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}