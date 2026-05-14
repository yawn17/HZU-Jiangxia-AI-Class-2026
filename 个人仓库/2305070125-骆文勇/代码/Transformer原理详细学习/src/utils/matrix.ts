import { Matrix } from '../types';

export function multiplyMatrices(A: Matrix, B: Matrix): Matrix {
  if (A.cols !== B.rows) {
    throw new Error('Matrix dimensions do not match for multiplication');
  }

  const result: number[][] = [];
  for (let i = 0; i < A.rows; i++) {
    result[i] = [];
    for (let j = 0; j < B.cols; j++) {
      let sum = 0;
      for (let k = 0; k < A.cols; k++) {
        sum += A.data[i][k] * B.data[k][j];
      }
      result[i][j] = sum;
    }
  }

  return {
    data: result,
    rows: A.rows,
    cols: B.cols,
    type: 'output'
  };
}

export function addMatrices(A: Matrix, B: Matrix): Matrix {
  if (A.rows !== B.rows || A.cols !== B.cols) {
    throw new Error('Matrix dimensions do not match for addition');
  }

  const result: number[][] = [];
  for (let i = 0; i < A.rows; i++) {
    result[i] = [];
    for (let j = 0; j < A.cols; j++) {
      result[i][j] = A.data[i][j] + B.data[i][j];
    }
  }

  return {
    data: result,
    rows: A.rows,
    cols: A.cols,
    type: 'output'
  };
}

export function transposeMatrix(A: Matrix): Matrix {
  const result: number[][] = [];
  for (let i = 0; i < A.cols; i++) {
    result[i] = [];
    for (let j = 0; j < A.rows; j++) {
      result[i][j] = A.data[j][i];
    }
  }

  return {
    data: result,
    rows: A.cols,
    cols: A.rows,
    type: A.type
  };
}

export function softmax(matrix: Matrix): Matrix {
  const result: number[][] = [];
  
  for (let i = 0; i < matrix.rows; i++) {
    const row = matrix.data[i];
    const max = Math.max(...row);
    const expRow = row.map(x => Math.exp(x - max));
    const sum = expRow.reduce((a, b) => a + b, 0);
    result[i] = expRow.map(x => x / sum);
  }

  return {
    data: result,
    rows: matrix.rows,
    cols: matrix.cols,
    type: 'attention'
  };
}

export function scaleMatrix(matrix: Matrix, factor: number): Matrix {
  const result: number[][] = [];
  for (let i = 0; i < matrix.rows; i++) {
    result[i] = [];
    for (let j = 0; j < matrix.cols; j++) {
      result[i][j] = matrix.data[i][j] * factor;
    }
  }

  return {
    data: result,
    rows: matrix.rows,
    cols: matrix.cols,
    type: matrix.type
  };
}

export function relu(matrix: Matrix): Matrix {
  const result: number[][] = [];
  for (let i = 0; i < matrix.rows; i++) {
    result[i] = [];
    for (let j = 0; j < matrix.cols; j++) {
      result[i][j] = Math.max(0, matrix.data[i][j]);
    }
  }

  return {
    data: result,
    rows: matrix.rows,
    cols: matrix.cols,
    type: matrix.type
  };
}

export function createRandomMatrix(rows: number, cols: number, type: string = 'output'): Matrix {
  const data: number[][] = [];
  for (let i = 0; i < rows; i++) {
    data[i] = [];
    for (let j = 0; j < cols; j++) {
      data[i][j] = (Math.random() - 0.5) * 2;
    }
  }

  return {
    data,
    rows,
    cols,
    type: type as any
  };
}

export function createZeroMatrix(rows: number, cols: number, type: string = 'output'): Matrix {
  const data: number[][] = [];
  for (let i = 0; i < rows; i++) {
    data[i] = new Array(cols).fill(0);
  }

  return {
    data,
    rows,
    cols,
    type: type as any
  };
}

export function normalizeMatrix(matrix: Matrix): Matrix {
  const result: number[][] = [];
  
  for (let i = 0; i < matrix.rows; i++) {
    const row = matrix.data[i];
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / row.length;
    const std = Math.sqrt(variance + 1e-8);
    
    result[i] = row.map(x => (x - mean) / std);
  }

  return {
    data: result,
    rows: matrix.rows,
    cols: matrix.cols,
    type: matrix.type
  };
}