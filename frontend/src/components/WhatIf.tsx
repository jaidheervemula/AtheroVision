import { TARGETS, type FeatureValue, type PatientFeatures, type PredictionResponse } from '../types';
import { pct, riskColor } from '../risk';
import { FEATURE_BY_KEY } from '../features';
import Alerts from './Alerts';

// What-if simulator: change a few modifiable risk factors and re-run the model on the edited
// copy of the patient. The parent owns the overrides and the scenario prediction.

interface Lever {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
}

const LEVERS: Lever[] = [
  { key: 'PR', label: 'Resting pulse', min: 40, max: 140, step: 1 },
  { key: 'LDL', label: 'LDL cholesterol', min: 40, max: 220, step: 5 },
  { key: 'HDL', label: 'HDL cholesterol', min: 20, max: 90, step: 1 },
  { key: 'BP', label: 'Systolic BP', min: 90, max: 200, step: 1 },
  { key: 'FBS', label: 'Fasting glucose', min: 70, max: 300, step: 5 },
  { key: 'Weight', label: 'Weight', min: 40, max: 150, step: 1 },
];

interface Props {
  features: PatientFeatures;
  overrides: Partial<PatientFeatures>;
  onOverride: (key: string, value: FeatureValue | undefined) => void;
  onReset: () => void;
  base: PredictionResponse | null;
  sim: PredictionResponse | null;
  showOnHeart: boolean;
  onShowOnHeart: (v: boolean) => void;
  scenario: PatientFeatures | null;
  ranges?: Record<string, [number, number]>;
}

export default function WhatIf({ features, overrides, onOverride, onReset, base, sim, showOnHeart, onShowOnHeart, scenario, ranges }: Props) {
  const changed = Object.keys(overrides).length > 0;
  const smoker = Number(features['Current Smoker']) === 1;
  const quit = overrides['Current Smoker'] === 0;

  const setQuit = (on: boolean) => {
    onOverride('Current Smoker', on ? 0 : undefined);
    onOverride('EX-Smoker', on ? 1 : undefined);
  };

  return (
    <section className="panel whatif" aria-label="What-if simulator">
      <header className="panel__head">
        <h2>What-if simulator</h2>
        <div className="whatif__actions">
          <label className="mini-toggle">
            <input type="checkbox" className="switch" checked={showOnHeart} disabled={!changed} onChange={(e) => onShowOnHeart(e.target.checked)} />
            <span>Show on heart</span>
          </label>
          <button className="chip" onClick={onReset} disabled={!changed}>Reset</button>
        </div>
      </header>

      <div className="levers">
        <label className={`lever lever--toggle${!smoker ? ' is-disabled' : ''}`}>
          <span>Quit smoking</span>
          <input type="checkbox" className="switch" checked={quit} disabled={!smoker} onChange={(e) => setQuit(e.target.checked)} />
          {!smoker && <em>not a current smoker</em>}
        </label>
        {LEVERS.map((l) => {
          const now = Number(features[l.key]);
          const val = overrides[l.key] !== undefined ? Number(overrides[l.key]) : now;
          const unit = FEATURE_BY_KEY[l.key]?.unit ?? '';
          const delta = val - now;
          return (
            <label key={l.key} className={`lever${overrides[l.key] !== undefined ? ' is-changed' : ''}`}>
              <span className="lever__top">
                <span>{l.label}</span>
                <span className="lever__val">
                  {val} <small>{unit}</small>
                  {delta !== 0 && <i className={delta < 0 ? 'down' : 'up'}>{delta > 0 ? '+' : ''}{delta}</i>}
                </span>
              </span>
              <input
                type="range"
                min={Math.min(l.min, now)}
                max={Math.max(l.max, now)}
                step={l.step}
                value={val}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  onOverride(l.key, v === now ? undefined : v);
                }}
                onDoubleClick={() => onOverride(l.key, undefined)}
                style={{ ['--pos' as string]: `${((val - Math.min(l.min, now)) / (Math.max(l.max, now) - Math.min(l.min, now))) * 100}%`, ['--now' as string]: `${((now - Math.min(l.min, now)) / (Math.max(l.max, now) - Math.min(l.min, now))) * 100}%` }}
              />
            </label>
          );
        })}
      </div>

      <table className="whatif__table">
        <thead>
          <tr><th /><th className="num">Now</th><th className="num">What-if</th><th className="num">Change</th></tr>
        </thead>
        <tbody>
          {TARGETS.map((t) => {
            const a = base?.predictions[t].probability;
            const b = sim?.predictions[t].probability ?? a;
            const d = a != null && b != null ? Math.round((b - a) * 100) : 0;
            return (
              <tr key={t}>
                <th scope="row">{t}</th>
                <td className="num" style={{ color: a != null ? riskColor(a) : undefined }}>{a != null ? pct(a) : '—'}</td>
                <td className="num" style={{ color: b != null ? riskColor(b) : undefined }}>{b != null ? pct(b) : '—'}</td>
                <td className={`num ${d < 0 ? 'down' : d > 0 ? 'up' : 'muted'}`}>{d === 0 ? '0' : `${d > 0 ? '+' : '−'}${Math.abs(d)} pts`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {scenario && <Alerts features={scenario} ranges={ranges} compact />}
      <p className="muted small whatif__note">
        Shows how the model's output responds to these inputs. It reflects associations in the training data,
        not the expected effect of a treatment.
      </p>
    </section>
  );
}
