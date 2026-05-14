import React, { createContext, useContext, useState, ReactNode } from 'react';
import { VisualizationState, Stage, AttentionStep, Token } from '../types';
import { TransformerSimulator } from '../utils/transformer';

interface VisualizationContextType {
  state: VisualizationState;
  setState: React.Dispatch<React.SetStateAction<VisualizationState>>;
  simulator: TransformerSimulator;
  updateProgress: (progress: number) => void;
  nextStage: () => void;
  prevStage: () => void;
  togglePlay: () => void;
  reset: () => void;
}

const VisualizationContext = createContext<VisualizationContextType | undefined>(undefined);

const stages: Stage[] = ['tokenization', 'embedding', 'positional-encoding', 'self-attention', 'add-norm', 'ffn'];
const attentionSteps: AttentionStep[] = ['linear-projection', 'attention-scores', 'scaling-masking', 'softmax', 'weighted-sum', 'multi-head'];

export function VisualizationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<VisualizationState>({
    currentStage: 'tokenization',
    attentionStep: 'linear-projection',
    inputText: 'The cat sat on the mat',
    tokens: [],
    isPlaying: false,
    progress: 0
  });

  const simulator = new TransformerSimulator();

  const updateProgress = (progress: number) => {
    setState(prev => ({ ...prev, progress: Math.max(0, Math.min(100, progress)) }));
  };

  const nextStage = () => {
    setState(prev => {
      if (prev.currentStage === 'self-attention') {
        const currentIndex = attentionSteps.indexOf(prev.attentionStep);
        if (currentIndex < attentionSteps.length - 1) {
          return { ...prev, attentionStep: attentionSteps[currentIndex + 1] };
        }
      }
      
      const currentIndex = stages.indexOf(prev.currentStage);
      if (currentIndex < stages.length - 1) {
        return { 
          ...prev, 
          currentStage: stages[currentIndex + 1],
          attentionStep: 'linear-projection'
        };
      }
      return prev;
    });
  };

  const prevStage = () => {
    setState(prev => {
      if (prev.currentStage === 'self-attention') {
        const currentIndex = attentionSteps.indexOf(prev.attentionStep);
        if (currentIndex > 0) {
          return { ...prev, attentionStep: attentionSteps[currentIndex - 1] };
        }
      }
      
      const currentIndex = stages.indexOf(prev.currentStage);
      if (currentIndex > 0) {
        return { 
          ...prev, 
          currentStage: stages[currentIndex - 1],
          attentionStep: 'linear-projection'
        };
      }
      return prev;
    });
  };

  const togglePlay = () => {
    setState(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
  };

  const reset = () => {
    setState({
      currentStage: 'tokenization',
      attentionStep: 'linear-projection',
      inputText: 'The cat sat on the mat',
      tokens: [],
      isPlaying: false,
      progress: 0
    });
  };

  return (
    <VisualizationContext.Provider value={{ 
      state, 
      setState, 
      simulator,
      updateProgress,
      nextStage,
      prevStage,
      togglePlay,
      reset
    }}>
      {children}
    </VisualizationContext.Provider>
  );
}

export function useVisualization() {
  const context = useContext(VisualizationContext);
  if (context === undefined) {
    throw new Error('useVisualization must be used within a VisualizationProvider');
  }
  return context;
}