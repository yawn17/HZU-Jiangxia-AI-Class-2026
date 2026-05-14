import React from 'react';
import { VisualizationProvider } from './context/VisualizationContext';
import { InputPanel } from './components/InputPanel/InputPanel';
import { VisualizationStage } from './components/VisualizationStage/VisualizationStage';
import { ControlPanel } from './components/ControlPanel/ControlPanel';

function App() {
  return (
    <VisualizationProvider>
      <div className="h-screen w-screen flex flex-col bg-monokai-bg overflow-hidden">
        <header className="h-16 bg-monokai-bg border-b-2 border-monokai-comment flex items-center px-6">
          <h1 className="text-2xl font-bold text-monokai-fg font-mono">
            Transformer 原理交互式可视化
          </h1>
          <span className="ml-4 text-monokai-comment font-mono text-sm">
            Self-Attention 演示
          </span>
        </header>
        
        <div className="flex-1 flex overflow-hidden">
          <InputPanel />
          <VisualizationStage />
          <ControlPanel />
        </div>
      </div>
    </VisualizationProvider>
  );
}

export default App;