import { FEATURE_BY_KEY } from '../features';
import type { PatientFeatures } from '../types';

const KEYS = ['BP', 'PR', 'EF-TTE', 'LDL', 'FBS', 'HDL'] as const;
const SHORT: Record<string, string> = { BP: 'Systolic BP', PR: 'Pulse', 'EF-TTE': 'Ejection fraction', LDL: 'LDL', FBS: 'Glucose', HDL: 'HDL' };

export default function Vitals({ features, compare }: { features: PatientFeatures; compare?: PatientFeatures }) {
  return (
    <div className="vitals" role="list" aria-label="Key vitals">
      {KEYS.map((k) => {
        const def = FEATURE_BY_KEY[k];
        const v = Number(features[k]);
        const [lo, hi] = def.normal ?? [-Infinity, Infinity];
        const state = v < lo ? 'low' : v > hi ? 'high' : 'ok';
        const d = compare ? Math.round((v - Number(compare[k])) * 10) / 10 : 0;
        return (
          <div key={k} role="listitem" className={`vital vital--${state}`}>
            <span className="vital__label">{SHORT[k]}</span>
            {/* keyed on value so the readout flashes when it changes */}
            <span key={v} className="vital__value">{Number.isInteger(v) ? v : v.toFixed(1)}</span>
            <span className="vital__unit">{def.unit}</span>
            {d !== 0 && <span className="vital__delta">{d > 0 ? '+' : '−'}{Math.abs(d)} vs now</span>}
          </div>
        );
      })}
    </div>
  );
}
