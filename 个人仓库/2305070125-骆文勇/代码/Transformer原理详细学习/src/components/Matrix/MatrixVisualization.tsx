import React from 'react';
import { motion } from 'framer-motion';
import { Matrix } from '../../types';
import { getColorForType, getValueColor } from '../../utils/colors';

interface MatrixProps {
  matrix: Matrix;
  title?: string;
  showValues?: boolean;
  highlightedCells?: { row: number; col: number }[];
  onCellHover?: (row: number, col: number) => void;
  onCellLeave?: () => void;
  maxDisplayRows?: number;
  maxDisplayCols?: number;
}

export function MatrixVisualization({ 
  matrix, 
  title, 
  showValues = false,
  highlightedCells = [],
  onCellHover,
  onCellLeave,
  maxDisplayRows = 8,
  maxDisplayCols = 8
}: MatrixProps) {
  const displayRows = Math.min(matrix.rows, maxDisplayRows);
  const displayCols = Math.min(matrix.cols, maxDisplayCols);
  const cellWidth = Math.max(30, Math.min(60, 400 / displayCols));
  const cellHeight = Math.max(30, Math.min(60, 300 / displayRows));

  const allValues = matrix.data.flat();
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const valueRange = maxValue - minValue || 1;

  const isHighlighted = (row: number, col: number) => {
    return highlightedCells.some(cell => cell.row === row && cell.col === col);
  };

  return (
    <div className="flex flex-col items-center gap-2">
      {title && (
        <h3 className="text-lg font-semibold text-monokai-fg font-mono">{title}</h3>
      )}
      <div className="flex items-center gap-2 text-xs text-monokai-comment font-mono">
        <span>{matrix.rows} × {matrix.cols}</span>
        {matrix.type && (
          <span className="px-2 py-1 rounded" style={{ backgroundColor: getColorForType(matrix.type) + '33' }}>
            {matrix.type}
          </span>
        )}
      </div>
      <div 
        className="border-2 border-monokai-comment rounded-lg overflow-hidden"
        style={{ 
          width: displayCols * cellWidth + 2,
          height: displayRows * cellHeight + 2
        }}
      >
        <div className="grid" style={{ 
          gridTemplateColumns: `repeat(${displayCols}, ${cellWidth}px)`,
          gridTemplateRows: `repeat(${displayRows}, ${cellHeight}px)`
        }}>
          {matrix.data.slice(0, displayRows).map((row, i) =>
            row.slice(0, displayCols).map((value, j) => {
              const normalizedValue = (value - minValue) / valueRange;
              const bgColor = getValueColor(value, minValue, maxValue);
              const isCellHighlighted = isHighlighted(i, j);
              
              return (
                <motion.div
                  key={`${i}-${j}`}
                  className="border border-monokai-comment flex items-center justify-center cursor-pointer transition-all"
                  style={{ 
                    backgroundColor: bgColor,
                    fontSize: Math.min(12, cellWidth / 3)
                  }}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ 
                    opacity: 1, 
                    scale: isCellHighlighted ? 1.1 : 1,
                    boxShadow: isCellHighlighted ? '0 0 10px rgba(249, 38, 114, 0.8)' : 'none'
                  }}
                  transition={{ delay: (i * displayCols + j) * 0.02 }}
                  onMouseEnter={() => onCellHover?.(i, j)}
                  onMouseLeave={onCellLeave}
                >
                  {showValues && (
                    <span className="font-mono font-bold" style={{ color: normalizedValue > 0.5 ? '#272822' : '#F8F8F2' }}>
                      {value.toFixed(2)}
                    </span>
                  )}
                </motion.div>
              );
            })
          )}
        </div>
      </div>
      {matrix.rows > maxDisplayRows || matrix.cols > maxDisplayCols && (
        <p className="text-xs text-monokai-comment font-mono">
          Showing {displayRows} × {displayCols} of {matrix.rows} × {matrix.cols}
        </p>
      )}
    </div>
  );
}