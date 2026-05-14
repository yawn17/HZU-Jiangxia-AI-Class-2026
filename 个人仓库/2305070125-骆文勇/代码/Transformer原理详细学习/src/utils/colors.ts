import { ColorScheme } from '../types';

export const monokaiColors: ColorScheme = {
  query: '#F92672',
  key: '#A6E22E',
  value: '#66D9EF',
  output: '#AE81FF',
  embedding: '#FD971F',
  positional: '#E6DB74',
  attention: '#F92672',
  weight: '#75715E',
  background: '#272822',
  foreground: '#F8F8F2'
};

export function getColorForType(type: string): string {
  const colors: Record<string, string> = {
    'input': '#F8F8F2',
    'query': monokaiColors.query,
    'key': monokaiColors.key,
    'value': monokaiColors.value,
    'output': monokaiColors.output,
    'embedding': monokaiColors.embedding,
    'positional': monokaiColors.positional,
    'attention': monokaiColors.attention,
    'weight': monokaiColors.weight
  };
  
  return colors[type] || '#F8F8F2';
}

export function getValueColor(value: number, min: number, max: number): string {
  const normalized = (value - min) / (max - min);
  const intensity = Math.max(0, Math.min(1, normalized));
  
  if (intensity < 0.5) {
    const r = Math.floor(39 + (249 - 39) * (intensity * 2));
    const g = Math.floor(40 + (38 - 40) * (intensity * 2));
    const b = Math.floor(34 + (114 - 34) * (intensity * 2));
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const r = Math.floor(249 + (249 - 249) * ((intensity - 0.5) * 2));
    const g = Math.floor(38 + (38 - 38) * ((intensity - 0.5) * 2));
    const b = Math.floor(114 + (114 - 114) * ((intensity - 0.5) * 2));
    return `rgb(${r}, ${g}, ${b})`;
  }
}

export function getHeatmapColor(value: number): string {
  const intensity = Math.max(0, Math.min(1, value));
  
  if (intensity < 0.33) {
    return `rgb(39, 40, 34)`;
  } else if (intensity < 0.66) {
    return `rgb(166, 226, 46)`;
  } else {
    return `rgb(249, 38, 114)`;
  }
}