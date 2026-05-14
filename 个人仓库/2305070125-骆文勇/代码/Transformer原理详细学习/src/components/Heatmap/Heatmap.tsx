import React from 'react';
import { motion } from 'framer-motion';
import { Matrix } from '../../types';
import { getHeatmapColor } from '../../utils/colors';

interface HeatmapProps {
  matrix: Matrix;
  title?: string;
  labels?: { rows?: string[]; cols?: string[] };
  highlightedCell?: { row: number; col: number };
  onCellHover?: (row: number, col: number) => void;
  onCellLeave?: () => void;
}

export function Heatmap({ 
  matrix, 
  title, 
  labels,
  highlightedCell,
  onCellHover,
  onCellLeave
}: HeatmapProps) {
  const cellSize = 50;
  const maxValue = Math.max(...matrix.data.flat());
  const minValue = Math.min(...matrix.data.flat());
  const valueRange = maxValue - minValue || 1;

  const getCellColor = (value: number) => {
    const normalized = (value - minValue) / valueRange;
    return getHeatmapColor(normalized);
  };

  const getTextColor = (value: number) => {
    const normalized = (value - minValue) / valueRange;
    return normalized > 0.5 ? '#272822' : '#F8F8F2';
  };

  const isHighlighted = (row: number, col: number) => {
    return highlightedCell?.row === row && highlightedCell?.col === col;
  };

  return (
    <div className="flex flex-col items-center gap-4">
      {title && (
        <h3 className="text-lg font-semibold text-monokai-fg font-mono">{title}</h3>
      )}
      <div className="flex flex-col items-center gap-2">
        {labels?.cols && (
          <div className="flex" style={{ marginLeft: cellSize }}>
            {labels.cols.map((label, i) => (
              <div 
                key={i} 
                className="text-xs text-monokai-comment font-mono flex items-center justify-center"
                style={{ width: cellSize }}
              >
                {label}
              </div>
            ))}
          </div>
        )}
        <div className="flex">
          {labels?.rows && (
            <div className="flex flex-col">
              {labels.rows.map((label, i) => (
                <div 
                  key={i} 
                  className="text-xs text-monokai-comment font-mono flex items-center justify-end pr-2"
                  style={{ height: cellSize }}
                >
                  {label}
                </div>
              ))}
            </div>
          )}
          <div 
            className="border-2 border-monokai-comment rounded-lg overflow-hidden"
            style={{ 
              width: matrix.cols * cellSize + 2,
              height: matrix.rows * cellSize + 2
            }}
          >
            <div className="grid" style={{ 
              gridTemplateColumns: `repeat(${matrix.cols}, ${cellSize}px)`,
              gridTemplateRows: `repeat(${matrix.rows}, ${cellSize}px)`
            }}>
              {matrix.data.map((row, i) =>
                row.map((value, j) => {
                  const cellHighlighted = isHighlighted(i, j);
                  
                  return (
                    <motion.div
                      key={`${i}-${j}`}
                      className="border border-monokai-comment flex items-center justify-center cursor-pointer relative"
                      style={{ 
                        backgroundColor: getCellColor(value),
                        color: getTextColor(value)
                      }}
                      initial={{ opacity: 0 }}
                      animate={{ 
                        opacity: 1,
                        scale: cellHighlighted ? 1.15 : 1,
                        zIndex: cellHighlighted ? 10 : 1,
                        boxShadow: cellHighlighted ? '0 0 15px rgba(249, 38, 114, 0.9)' : 'none'
                      }}
                      transition={{ delay: (i * matrix.cols + j) * 0.03 }}
                      onMouseEnter={() => onCellHover?.(i, j)}
                      onMouseLeave={onCellLeave}
                    >
                      <span className="font-mono font-bold text-sm">
                        {value.toFixed(3)}
                      </span>
                    </motion.div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2 text-xs text-monokai-comment font-mono">
        <span>Low</span>
        <div className="w-32 h-3 rounded" style={{ 
          background: `linear-gradient(to right, rgb(39, 40, 34), rgb(166, 226, 46), rgb(249, 38, 114))`
        }} />
        <span>High</span>
      </div>
    </div>
  );
}