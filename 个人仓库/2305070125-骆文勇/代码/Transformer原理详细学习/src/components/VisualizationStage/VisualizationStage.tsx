import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useVisualization } from '../../context/VisualizationContext';
import { MatrixVisualization } from '../Matrix/MatrixVisualization';
import { Heatmap } from '../Heatmap/Heatmap';
import { Tokenization } from '../../stages/Tokenization';
import { Embedding } from '../../stages/Embedding';
import { PositionalEncoding } from '../../stages/PositionalEncoding';
import { SelfAttention } from '../../stages/SelfAttention';
import { AddNorm } from '../../stages/AddNorm';
import { FFN } from '../../stages/FFN';

export function VisualizationStage() {
  const { state, simulator } = useVisualization();
  const [tokens, setTokens] = useState<any[]>([]);

  useEffect(() => {
    const newTokens = simulator.tokenize(state.inputText);
    setTokens(newTokens);
  }, [state.inputText, simulator]);

  return (
    <div className="flex-1 bg-monokai-bg p-6 flex flex-col">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-monokai-fg font-mono">
          Transformer 原理可视化
        </h1>
        <p className="text-monokai-comment font-mono text-sm mt-1">
          当前阶段: {state.currentStage}
          {state.currentStage === 'self-attention' && ` - ${state.attentionStep}`}
        </p>
      </div>

      <div className="flex-1 flex items-center justify-center overflow-auto">
        <AnimatePresence mode="wait">
          {state.currentStage === 'tokenization' && (
            <motion.div
              key="tokenization"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="w-full h-full flex items-center justify-center"
            >
              <Tokenization tokens={tokens} />
            </motion.div>
          )}

          {state.currentStage === 'embedding' && (
            <motion.div
              key="embedding"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="w-full h-full flex items-center justify-center"
            >
              <Embedding tokens={tokens} simulator={simulator} />
            </motion.div>
          )}

          {state.currentStage === 'positional-encoding' && (
            <motion.div
              key="positional-encoding"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="w-full h-full flex items-center justify-center"
            >
              <PositionalEncoding tokens={tokens} simulator={simulator} />
            </motion.div>
          )}

          {state.currentStage === 'self-attention' && (
            <motion.div
              key="self-attention"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="w-full h-full flex items-center justify-center"
            >
              <SelfAttention tokens={tokens} simulator={simulator} attentionStep={state.attentionStep} />
            </motion.div>
          )}

          {state.currentStage === 'add-norm' && (
            <motion.div
              key="add-norm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="w-full h-full flex items-center justify-center"
            >
              <AddNorm tokens={tokens} simulator={simulator} />
            </motion.div>
          )}

          {state.currentStage === 'ffn' && (
            <motion.div
              key="ffn"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="w-full h-full flex items-center justify-center"
            >
              <FFN tokens={tokens} simulator={simulator} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}