import { useEffect, useState } from 'react';
import { fetchMetrics } from '../api/client';
import { TARGETS, type MetricsResponse } from '../types';
import CalibrationPlot from './CalibrationPlot';

const COLS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] as const;
const HEAD: Record<(typeof COLS)[number], [string, string]> = {
  accuracy: ['Acc', 'Accuracy'],
  precision: ['Prec', 'Precision'],
  recall: ['Rec', 'Recall (sensitivity)'],
  f1: ['F1', 'F1 score'],
  roc_auc: ['AUC', 'ROC-AUC'],
};

/** Short, readable model names for the table (e.g. "logreg_clinical" -> "Logistic, clinical"). */
function modelLabel(name: string, n: number) {
  const kind = name.startsWith('logreg') ? 'Logistic' : name.startsWith('rf') ? 'Random forest' : name;
  return `${kind}, ${n} inputs`;
}

export default function MetricsCard() {
  const [m, setM] = useState<MetricsResponse | null>(null);
  useEffect(() => { fetchMetrics().then(setM); }, []);

  return (
    <section className="panel metrics" aria-label="Model performance">
      <header className="panel__head">
        <h2>Model performance</h2>
        <span className="muted">{m ? `${m.validation}, ${m.n_samples} patients` : 'Appears when the model service is connected'}</span>
      </header>
      <table>
        <thead>
          <tr><th />{COLS.map((c) => <th key={c} className="num" title={HEAD[c][1]}>{HEAD[c][0]}</th>)}</tr>
        </thead>
        <tbody>
          {TARGETS.map((t) => {
            const info = m?.models?.[t];
            return (
              <tr key={t}>
                <th scope="row">
                  {t}
                  {info && <span className="metrics__model">{modelLabel(info.name, info.n_input_features)}</span>}
                </th>
                {COLS.map((c) => {
                  const sd = m?.sd?.[t]?.[c];
                  return (
                    <td key={c} className="num" title={m && sd != null ? `${HEAD[c][1]} ${m.targets[t][c].toFixed(3)} ± ${sd.toFixed(3)} (mean ± SD across folds)` : undefined}>
                      {m ? m.targets[t][c].toFixed(2) : '—'}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      {m && (
        <p className="metrics__note">
          Mean across cross-validation folds at a 0.5 threshold (LAD: the deployed calibrated model's estimate). Hover a value for its spread.
          No external validation.
        </p>
      )}
      {m?.operating_points && Object.values(m.operating_points).some((o) => o?.mode === 'screening') && (
        <div className="metrics__cut">
          <p className="eyebrow">Flagging cut-offs</p>
          <table>
            <thead><tr><th /><th className="num">Cut-off</th><th className="num" title="Share of diseased patients caught (cross-validated)">Catches</th><th className="num" title="Share of healthy patients correctly cleared (cross-validated)">Clears</th></tr></thead>
            <tbody>
              {TARGETS.map((t) => {
                const o = m.operating_points![t];
                if (!o || !o.cv) return null;
                const was = o.mode === 'screening' && o.cv_at_default ? o.cv_at_default : null;
                return (
                  <tr key={t}>
                    <th scope="row">{t}</th>
                    <td className="num">{Math.round(o.threshold * 100)}%</td>
                    <td className="num">{Math.round(o.cv.sensitivity * 100)}%{was && <span className="muted"> (was {Math.round(was.sensitivity * 100)}%)</span>}</td>
                    <td className="num">{Math.round(o.cv.specificity * 100)}%{was && <span className="muted"> (was {Math.round(was.specificity * 100)}%)</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {m.operating_rule && <p className="metrics__note">{m.operating_rule}</p>}
        </div>
      )}
      {m?.calibration && Object.keys(m.calibration).length > 0 && (
        <div className="metrics__calib">
          <p className="eyebrow">Risk estimation quality: predicted vs observed</p>
          <div className="calib-grid">
            {TARGETS.map((t) => m.calibration![t] && <CalibrationPlot key={t} target={t} info={m.calibration![t]!} />)}
          </div>
          {m.calibration_note && <p className="metrics__note">{m.calibration_note} Slope 1 is ideal; below 1, probabilities are too extreme; above 1, too cautious.</p>}
          <details className="calib-table">
            <summary>Show as table</summary>
            <table>
              <thead><tr><th>Target</th><th className="num">Brier</th><th className="num">Slope (95% CI)</th></tr></thead>
              <tbody>
                {TARGETS.map((t) => {
                  const c = m.calibration![t];
                  if (!c) return null;
                  return (
                    <tr key={t}>
                      <th scope="row">{t}</th>
                      <td className="num">{c.brier ? c.brier.estimate.toFixed(3) : '—'}</td>
                      <td className="num">{c.calibration_slope ? `${c.calibration_slope.estimate.toFixed(2)} (${c.calibration_slope.ci_low.toFixed(2)}–${c.calibration_slope.ci_high.toFixed(2)})` : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {TARGETS.map((t) => m.calibration![t] && (
              <table key={t} className="calib-table__bins">
                <caption>{t}: groups of patients</caption>
                <thead><tr><th className="num">Predicted</th><th className="num">Observed</th><th className="num">95% CI</th><th className="num">n</th></tr></thead>
                <tbody>
                  {m.calibration![t]!.bins.map((b, i) => (
                    <tr key={i}>
                      <td className="num">{Math.round(b.mean_predicted * 100)}%</td>
                      <td className="num">{Math.round(b.observed * 100)}%</td>
                      <td className="num">{Math.round(b.ci_low * 100)}–{Math.round(b.ci_high * 100)}%</td>
                      <td className="num">{b.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ))}
          </details>
        </div>
      )}
    </section>
  );
}
