
// One colour scale used everywhere (3D vessels, bars, badges) so they always agree.
const STOPS: [number, string][] = [
  [0.0, '#5fe0a0'],
  [0.35, '#b9d65a'],
  [0.5, '#f2c14e'],
  [0.72, '#f28a3c'],
  [1.0, '#ff5a6e'],
];

// Plain hex interpolation (kept free of three.js so the dashboard bundle stays small).
function mixHex(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return '#' + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, '0')).join('');
}

export function riskColor(p: number): string {
  const x = Math.min(1, Math.max(0, p));
  for (let i = 1; i < STOPS.length; i++) {
    const [p1, c1] = STOPS[i];
    if (x <= p1) {
      const [p0, c0] = STOPS[i - 1];
      const t = (x - p0) / (p1 - p0);
      return mixHex(c0, c1, t);
    }
  }
  return STOPS[STOPS.length - 1][1];
}

export function riskBand(p: number): 'Low' | 'Moderate' | 'High' {
  return p < 0.35 ? 'Low' : p < 0.65 ? 'Moderate' : 'High';
}

export const pct = (p: number) => `${Math.round(p * 100)}%`;
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export const VESSEL_INFO = {
  LAD: { name: 'Left Anterior Descending', supplies: 'Front of the left ventricle, apex and interventricular septum' },
  LCX: { name: 'Left Circumflex', supplies: 'Lateral and posterior walls of the left ventricle' },
  RCA: { name: 'Right Coronary Artery', supplies: 'Right ventricle and inferior wall of the left ventricle' },
} as const;
