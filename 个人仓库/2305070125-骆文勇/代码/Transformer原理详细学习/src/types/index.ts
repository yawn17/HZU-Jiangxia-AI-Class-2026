export type Stage = 
  | 'tokenization'
  | 'embedding'
  | 'positional-encoding'
  | 'self-attention'
  | 'add-norm'
  | 'ffn';

export type AttentionStep = 
  | 'linear-projection'
  | 'attention-scores'
  | 'scaling-masking'
  | 'softmax'
  | 'weighted-sum'
  | 'multi-head';

export type MatrixType = 
  | 'input'
  | 'query'
  | 'key'
  | 'value'
  | 'output'
  | 'embedding'
  | 'positional'
  | 'attention'
  | 'weight';

export interface Matrix {
  data: number[][];
  rows: number;
  cols: number;
  type: MatrixType;
  label?: string;
}

export interface Token {
  text: string;
  id: number;
}

export interface VisualizationState {
  currentStage: Stage;
  attentionStep: AttentionStep;
  inputText: string;
  tokens: Token[];
  isPlaying: boolean;
  progress: number;
  selectedCell?: { row: number; col: number };
}

export interface ColorScheme {
  query: string;
  key: string;
  value: string;
  output: string;
  embedding: string;
  positional: string;
  attention: string;
  weight: string;
  background: string;
  foreground: string;
}