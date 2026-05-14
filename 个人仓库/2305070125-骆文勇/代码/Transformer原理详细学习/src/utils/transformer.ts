import { Matrix, Token } from '../types';
import { 
  multiplyMatrices, 
  addMatrices, 
  transposeMatrix, 
  softmax, 
  scaleMatrix, 
  createRandomMatrix,
  createZeroMatrix,
  relu
} from './matrix';

export class TransformerSimulator {
  private vocabSize: number = 10000;
  private dModel: number = 512;
  private dK: number = 64;
  private numHeads: number = 8;
  private embeddingMatrix: Matrix;
  private WQ: Matrix;
  private WK: Matrix;
  private WV: Matrix;
  private WO: Matrix;

  constructor() {
    this.embeddingMatrix = createRandomMatrix(this.vocabSize, this.dModel, 'embedding');
    this.WQ = createRandomMatrix(this.dModel, this.dK, 'weight');
    this.WK = createRandomMatrix(this.dModel, this.dK, 'weight');
    this.WV = createRandomMatrix(this.dModel, this.dK, 'weight');
    this.WO = createRandomMatrix(this.dModel, this.dModel, 'weight');
  }

  tokenize(text: string): Token[] {
    const words = text.toLowerCase().split(/\s+/);
    return words.map((word, index) => ({
      text: word,
      id: Math.floor(Math.random() * this.vocabSize)
    }));
  }

  getWQ(): Matrix {
    return this.WQ;
  }

  getWK(): Matrix {
    return this.WK;
  }

  getWV(): Matrix {
    return this.WV;
  }

  embeddingLookup(tokens: Token[]): Matrix {
    const seqLen = tokens.length;
    const data: number[][] = [];
    
    for (let i = 0; i < seqLen; i++) {
      const tokenId = tokens[i].id % this.vocabSize;
      data[i] = [...this.embeddingMatrix.data[tokenId]];
    }

    return {
      data,
      rows: seqLen,
      cols: this.dModel,
      type: 'embedding'
    };
  }

  positionalEncoding(seqLen: number): Matrix {
    const data: number[][] = [];
    
    for (let pos = 0; pos < seqLen; pos++) {
      data[pos] = [];
      for (let i = 0; i < this.dModel; i++) {
        if (i % 2 === 0) {
          data[pos][i] = Math.sin(pos / Math.pow(10000, i / this.dModel));
        } else {
          data[pos][i] = Math.cos(pos / Math.pow(10000, (i - 1) / this.dModel));
        }
      }
    }

    return {
      data,
      rows: seqLen,
      cols: this.dModel,
      type: 'positional'
    };
  }

  linearProjection(X: Matrix): { Q: Matrix; K: Matrix; V: Matrix } {
    const Q = multiplyMatrices(X, this.WQ);
    const K = multiplyMatrices(X, this.WK);
    const V = multiplyMatrices(X, this.WV);

    Q.type = 'query';
    K.type = 'key';
    V.type = 'value';

    return { Q, K, V };
  }

  computeAttentionScores(Q: Matrix, K: Matrix): Matrix {
    const KT = transposeMatrix(K);
    const scores = multiplyMatrices(Q, KT);
    scores.type = 'attention';
    return scores;
  }

  scaleAttentionScores(scores: Matrix): Matrix {
    return scaleMatrix(scores, 1 / Math.sqrt(this.dK));
  }

  applySoftmax(scores: Matrix): Matrix {
    return softmax(scores);
  }

  weightedSum(attentionWeights: Matrix, V: Matrix): Matrix {
    return multiplyMatrices(attentionWeights, V);
  }

  multiHeadAttention(X: Matrix): Matrix {
    const heads: Matrix[] = [];
    
    for (let h = 0; h < this.numHeads; h++) {
      const { Q, K, V } = this.linearProjection(X);
      const scores = this.computeAttentionScores(Q, K);
      const scaledScores = this.scaleAttentionScores(scores);
      const attentionWeights = this.applySoftmax(scaledScores);
      const headOutput = this.weightedSum(attentionWeights, V);
      heads.push(headOutput);
    }

    const concatenated = this.concatenateHeads(heads);
    const output = multiplyMatrices(concatenated, this.WO);
    output.type = 'output';
    
    return output;
  }

  private concatenateHeads(heads: Matrix[]): Matrix {
    const seqLen = heads[0].rows;
    const dModel = heads[0].cols * this.numHeads;
    const data: number[][] = [];

    for (let i = 0; i < seqLen; i++) {
      data[i] = [];
      for (let h = 0; h < this.numHeads; h++) {
        data[i] = [...data[i], ...heads[h].data[i]];
      }
    }

    return {
      data,
      rows: seqLen,
      cols: dModel,
      type: 'output'
    };
  }

  addAndNorm(X: Matrix, attentionOutput: Matrix): Matrix {
    const added = addMatrices(X, attentionOutput);
    return this.layerNorm(added);
  }

  private layerNorm(X: Matrix): Matrix {
    const data: number[][] = [];
    
    for (let i = 0; i < X.rows; i++) {
      const row = X.data[i];
      const mean = row.reduce((a, b) => a + b, 0) / row.length;
      const variance = row.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / row.length;
      const std = Math.sqrt(variance + 1e-8);
      
      data[i] = row.map(x => (x - mean) / std);
    }

    return {
      data,
      rows: X.rows,
      cols: X.cols,
      type: X.type
    };
  }

  feedForward(X: Matrix): Matrix {
    const dFF = this.dModel * 4;
    const W1 = createRandomMatrix(this.dModel, dFF, 'weight');
    const W2 = createRandomMatrix(dFF, this.dModel, 'weight');
    
    const hidden = multiplyMatrices(X, W1);
    const activated = relu(hidden);
    const output = multiplyMatrices(activated, W2);
    output.type = 'output';
    
    return output;
  }
}