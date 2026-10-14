import { useState } from 'react';
import { TARGETS, type PredictionResponse, type Target } from '../types';
import { labelOf } from '../features';
import { pct, sigmoid } from '../risk';
import { formatValue } from './format';

interface Props {
  result: PredictionResponse | null;
  target: Target;
  onTarget: (t: Target) => void;
  /** A newer explanation is being calculated; the one shown belongs to the previous inputs. */
  explaining?: boolean;
}

const TOP_N = 10;

export default function ContributionChart({ result, target, onTarget, explaining = false }: Props) {
  const [showAll, setShowAll] = useState(false);
  if (!result) return null;
  const pred = result.predictions[target];
  const prob = pred.units === 'probability';
  const stale = result.explained === false;
  const scale = prob ? 100 : 1; // probability impacts are shown in percentage points
  const unit = prob ? 'pts' : 'log-odds';
  const fmt = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v * scale).toFixed(prob ? 1 : 2)}`;

  const sorted = [...pred.contributions].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  const shown = showAll ? sorted.filter((c) => Math.abs(c.shap) > 1e-6) : sorted.slice(0, TOP_N);
  const rest = sorted.slice(shown.length);
  const restSum = rest.reduce((a, c) => a + c.shap, 0);
  const max = Math.max(...shown.map((c) => Math.abs(c.shap)), Math.abs(restSum), prob ? 1e-4 : 0.01);
  const reference = pred.reference_probability ?? sigmoid(pred.base_value);
  const notUsed = pred.features_not_used?.length ?? 0;

  return (
    <section className={`panel contrib${stale ? ' contrib--stale' : ''}`} aria-label="Feature contributions" aria-busy={explaining}>
      <header className="panel__head">
        <h2>Why this prediction</h2>
        <div className="tabs" role="tablist">
          {TARGETS.map((t) => (
            <button key={t} role="tab" aria-selected={t === target} className={t === target ? 'on' : ''} onClick={() => onTarget(t)}>
              {t}
            </button>
          ))}
        </div>
      </header>

      {sorted.length === 0 ? (
        <p className="contrib__empty">{explaining ? 'Calculating the explanation…' : 'No explanation available.'}</p>
      ) : (
        <>
          <p className="contrib__summary">
            Reference patient <b>{pct(reference)}</b>
            <span className="arrow">→</span>
            this patient <b>{pct(pred.probability)}</b>
            {stale && <span className="contrib__updating" role="status">Updating explanation…</span>}
          </p>
          <p className="contrib__method">
            <span className={`method-badge${prob ? ' method-badge--approx' : ''}`}>
              {prob ? 'Approximate impact · not SHAP' : 'SHAP values · exact'}
            </span>{' '}
            {prob
              ? 'Change in probability (percentage points) when each input alone is replaced by values from the training data. Random forest: these do not add up exactly to the score.'
              : 'For this logistic-regression model, the SHAP value of each input is computed exactly in log-odds (inputs treated as independent). Together they add up to this patient’s score.'}
            {pred.model_name && <> Model: {pred.model_name}{pred.n_input_features ? `, ${pred.n_input_features} inputs` : ''}.</>}
            {' '}<b>{result.source === 'mock' ? 'Demo model: contributions are illustrative only.' : 'Contributions are statistical associations learned from 303 patients, not causes.'}</b>
          </p>

          <div className="legend">
            <span><i className="sw sw--down" />lowers risk</span>
            <span><i className="sw sw--up" />raises risk</span>
          </div>

          <ol className="shap">
            {shown.map((c) => (
              <li key={c.feature} className="shap__row" title={`${labelOf(c.feature)}: ${fmt(c.shap)} ${unit}`}>
                <span className="shap__label">
                  {labelOf(c.feature)}
                  <span className="shap__value">{formatValue(c.feature, c.value)}</span>
                </span>
                <span className="shap__track">
                  <span className="shap__axis" />
                  <span className={`shap__bar ${c.shap >= 0 ? 'up' : 'down'}`} style={{ width: `${(Math.abs(c.shap) / max) * 50}%` }} />
                </span>
                <span className={`shap__num ${c.shap >= 0 ? 'up' : 'down'}`}>{fmt(c.shap)}</span>
              </li>
            ))}
            {rest.length > 0 && Math.abs(restSum * scale) > 1e-3 && (
              <li className="shap__row shap__row--rest">
                <span className="shap__label">{rest.length} other inputs</span>
                <span className="shap__track">
                  <span className="shap__axis" />
                  <span className={`shap__bar ${restSum >= 0 ? 'up' : 'down'}`} style={{ width: `${(Math.abs(restSum) / max) * 50}%` }} />
                </span>
                <span className={`shap__num ${restSum >= 0 ? 'up' : 'down'}`}>{fmt(restSum)}</span>
              </li>
            )}
          </ol>
          {notUsed > 0 && (
            <p className="contrib__notused">
              This model does not use the other {notUsed} inputs; changing them does not affect its score.
            </p>
          )}
          {sorted.length > TOP_N && (
            <button className="link" onClick={() => setShowAll((s) => !s)}>
              {showAll ? `Show top ${TOP_N}` : 'Show all contributing inputs'}
            </button>
          )}
        </>
      )}
    </section>
  );
}
