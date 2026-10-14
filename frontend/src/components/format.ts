import { FEATURE_BY_KEY } from '../features';
import type { FeatureValue } from '../types';

export function formatValue(key: string, v: FeatureValue): string {
  const def = FEATURE_BY_KEY[key];
  if (!def) return String(v);
  if (def.kind === 'bool') return Number(v) === 1 ? 'Yes' : 'No';
  if (def.kind === 'select') return v === 'N' ? 'None' : String(v);
  const num = Number(v);
  const txt = Number.isInteger(num) ? num.toLocaleString() : num.toFixed(1);
  return def.unit ? `${txt} ${def.unit}` : txt;
}
