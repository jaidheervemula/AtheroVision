import { FEATURES } from '../features';
import type { PatientFeatures, PredictionResponse, Target } from '../types';
import { formatValue } from './format';

interface Props {
  features: PatientFeatures;
  result: PredictionResponse | null;
  target: Target;
}

// Physiological measurements (numeric vitals, labs, echo) with reference ranges and their
// share of the model's total attribution for the selected target.
export default function MeasurementsTable({ features, result, target }: Props) {
  if (!result) return null;
  const contribs = result.predictions[target].contributions;
  // An input the model doesn't use (or never saw, e.g. height) has no contribution at all.
  const used = new Set(contribs.map((c) => c.feature));
  const byKey = Object.fromEntries(contribs.map((c) => [c.feature, c.shap]));
  const total = contribs.reduce((a, c) => a + Math.abs(c.shap), 0) || 1;

  const rows = FEATURES.filter((f) => f.kind === 'number' && f.normal)
    .map((f) => {
      const v = Number(features[f.key]);
      const status = v < f.normal![0] ? 'Low' : v > f.normal![1] ? 'High' : 'Normal';
      const shap = byKey[f.key] ?? 0;
      return { f, v, status, shap, share: Math.abs(shap) / total, isUsed: used.has(f.key) };
    })
    .sort((a, b) => Number(b.isUsed) - Number(a.isUsed) || b.share - a.share);

  return (
    <section className="panel measures" aria-label="Physiological measurements">
      <header className="panel__head">
        <h2>Physiological measurements</h2>
        <span className="muted">Share of {target} attribution</span>
      </header>
      {contribs.length > 0 && !rows.some((r) => r.isUsed) && (
        <p className="measures__none">
          The {target} model uses none of these measurements; its {result.predictions[target].n_input_features ?? 'few'} inputs
          are history and symptoms. Values are shown for reference.
        </p>
      )}
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Measurement</th><th className="num">Value</th><th>Status</th><th>Share</th></tr>
          </thead>
          <tbody>
            {rows.map(({ f, v, status, shap, share, isUsed }) => (
              <tr key={f.key}>
                <td>{f.label}</td>
                <td className="num">
                  {formatValue(f.key, v)}
                  <span className="ref">ref {f.normal![0]}–{f.normal![1]}</span>
                </td>
                <td><span className={`pill pill--${status.toLowerCase()}`}>{status}</span></td>
                <td>
                  {!contribs.length ? <span className="muted">…</span> : isUsed ? (
                    <>
                      <span className="mini">
                        <span className={`mini__bar ${shap >= 0 ? 'up' : 'down'}`} style={{ width: `${Math.min(100, share * 300)}%` }} />
                      </span>
                      <span className={`mini__num ${shap >= 0 ? 'up' : 'down'}`}>{Math.round(share * 100)}%</span>
                    </>
                  ) : <span className="muted small" title={`The ${target} model does not use this input`}>Not used</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
