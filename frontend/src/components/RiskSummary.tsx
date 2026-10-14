import { TARGETS, VESSELS, type DatasetPatient, type PredictionResponse, type Target } from '../types';
import { VESSEL_INFO, pct, riskBand, riskColor } from '../risk';
import { labelOf } from '../features';
import { assessmentSummary } from '../summary';

interface Props {
  result: PredictionResponse | null;
  selected: Target;
  onSelect: (t: Target) => void;
  /** Set when the case is a real dataset patient: the angiography outcome to compare against. */
  truth?: DatasetPatient | null;
}

const WHAT: Record<Target, string> = {
  CAD: 'coronary artery disease', LAD: 'LAD stenosis', LCX: 'LCX stenosis', RCA: 'RCA stenosis',
};

/** "Of the 40 patients whose held-out score was closest to this one, 12 actually had LCX stenosis." */
function RealityCheckNote({ target, result }: { target: Target; result: PredictionResponse }) {
  const rc = result.predictions[target].reality_check;
  if (!rc) return null;
  const range = `${pct(rc.score_min)}–${pct(rc.score_max)}`;
  return (
    <div className="reality" aria-label={`Reality check for ${target}`}>
      <p className="eyebrow">Reality check · {target}</p>
      <div className="reality__body">
        <span className="reality__big">{rc.positives}<span className="muted">/{rc.n}</span></span>
        <p>
          Of the {rc.n} patients in our data whose held-out {target} score was closest to this one ({range}),{' '}
          <b>{rc.positives} actually had {WHAT[target]}</b> on angiography: {pct(rc.rate)} (95% CI {pct(rc.ci_low)}–{pct(rc.ci_high)}).
          {rc.excluded_row != null && ' This patient is left out of their own comparison.'}
        </p>
      </div>
      <div className="reality__dots" aria-hidden>
        {Array.from({ length: rc.n }, (_, i) => <span key={i} className={i < rc.positives ? 'on' : ''} />)}
      </div>
    </div>
  );
}

const OUTCOME: Record<Target, [string, string]> = {
  CAD: ['No CAD', 'CAD'], LAD: ['Normal', 'Stenotic'], LCX: ['Normal', 'Stenotic'], RCA: ['Normal', 'Stenotic'],
};

function TruthTable({ truth, result }: { truth: DatasetPatient; result: PredictionResponse }) {
  return (
    <div className="truth" aria-label="Comparison with the angiography result">
      <div className="truth__head">
        <h3 className="eyebrow">Dataset row {truth.row}: actual result</h3>
      </div>
      {truth.label_conflict && (
        <p className="truth__conflict" role="note"><b>⚠ Label conflict in dataset.</b> {truth.label_conflict.replace(/^Label conflict in the dataset: (\w)/, (_, c: string) => c.toUpperCase())}</p>
      )}
      <table>
        <thead>
          <tr><th>Target</th><th>Angiography</th><th className="num">Live</th><th className="num">Cross-val.</th></tr>
        </thead>
        <tbody>
          {TARGETS.map((t) => {
            const actual = truth.actual[t];
            const cv = truth.cv[t];
            const cut = result.predictions[t].threshold;
            const cvRight = cv ? (cv.probability >= cut ? 1 : 0) === actual : null;
            return (
              <tr key={t}>
                <td><b>{t}</b></td>
                <td>{OUTCOME[t][actual]}{t === 'CAD' && truth.label_conflict && <span className="truth__flag" title="This label contradicts the vessel labels"> ⚠</span>}</td>
                <td className="num">{pct(result.predictions[t].probability)}</td>
                <td className={`num ${cvRight == null ? '' : cvRight ? 'hit' : 'miss'}`}
                  title={cv ? `Range over ${cv.repeats} repeats: ${pct(cv.min)}–${pct(cv.max)}` : 'Not available'}>
                  {cv ? `${pct(cv.probability)} ${cvRight ? '✓' : '✗'}` : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p>{truth.note} ✓/✗: whether the cross-validated prediction, at each target's flagging cut-off, matches angiography.</p>
    </div>
  );
}

/** Probability bar; the tick marks the cut-off at which this target is flagged. */
function Bar({ p, cut = 0.5 }: { p: number; cut?: number }) {
  return (
    <div className="bar" aria-hidden>
      <div className="bar__fill" style={{ width: pct(p), background: riskColor(p) }} />
      <div className="bar__tick" style={{ left: `${cut * 100}%` }} />
    </div>
  );
}

export default function RiskSummary({ result, selected, onSelect, truth }: Props) {
  if (!result) return <section className="panel"><p className="muted">Waiting for prediction…</p></section>;
  const cad = result.predictions.CAD;
  const vessel = selected === 'CAD' ? null : selected;

  return (
    <section className="panel risk" aria-label="Predicted risk">
      <button
        className={`cad${selected === 'CAD' ? ' is-selected' : ''}`}
        style={{ ['--risk' as string]: riskColor(cad.probability) }}
        onClick={() => onSelect('CAD')}
      >
        <span className="eyebrow">Overall CAD</span>
        <span className="cad__value">{pct(cad.probability)}</span>
        <span className="cad__status">
          {cad.label ? 'CAD predicted' : 'No CAD predicted'}, {riskBand(cad.probability).toLowerCase()} risk
        </span>
        <Bar p={cad.probability} cut={cad.threshold} />
      </button>

      <div className="summary" aria-label="Assessment summary" aria-live="polite">
        {assessmentSummary(result).map((line) => <p key={line}>{line}</p>)}
      </div>

      {result.agreement?.models_disagree && (
        <p className="disagree" role="note">
          <b>Models disagree.</b> {result.agreement.note}
        </p>
      )}

      <h3 className="eyebrow">Vessel stenosis (≥50% narrowing)</h3>
      <ul className="vessels">
        {VESSELS.map((v) => {
          const pr = result.predictions[v];
          const p = pr.probability;
          return (
            <li key={v}>
              <button
                className={`vessel${selected === v ? ' is-selected' : ''}`}
                style={{ ['--risk' as string]: riskColor(p) }}
                onClick={() => onSelect(selected === v ? 'CAD' : v)}
              >
                <span className="vessel__name">
                  <b>{v}</b>
                  <span className="muted">
                    {VESSEL_INFO[v].name}
                    {pr.operating_point?.mode === 'screening' && <span className="vessel__cut" title={pr.operating_point.reason}> · flagged from {pct(pr.threshold)}</span>}
                  </span>
                </span>
                <span className="vessel__pct">{pct(p)}{pr.label ? <span className="vessel__flag" title={`At or above the ${pct(pr.threshold)} cut-off`}>flagged</span> : null}</span>
                <Bar p={p} cut={pr.threshold} />
              </button>
            </li>
          );
        })}
      </ul>

      <RealityCheckNote target={selected} result={result} />

      {truth && <TruthTable truth={truth} result={result} />}

      {vessel && (
        <div className="detail" style={{ ['--risk' as string]: riskColor(result.predictions[vessel].probability) }}>
          <h3>{VESSEL_INFO[vessel].name}</h3>
          <p className="muted">Supplies: {VESSEL_INFO[vessel].supplies}. The tinted region on the heart shows this territory.</p>
          <p>
            <b>{pct(result.predictions[vessel].probability)}</b> probability of significant stenosis
            ({result.predictions[vessel].label ? 'flagged' : 'not flagged'} at the {pct(result.predictions[vessel].threshold)} cut-off).
          </p>
          {result.predictions[vessel].operating_point?.mode === 'screening' && (
            <p className="muted small">{result.predictions[vessel].operating_point!.reason}</p>
          )}
          <p className="eyebrow">Top drivers</p>
          <ul className="drivers">
            {[...result.predictions[vessel].contributions]
              .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
              .slice(0, 4)
              .map((c) => (
                <li key={c.feature}>
                  <span className={c.shap > 0 ? 'up' : 'down'}>{c.shap > 0 ? '▲' : '▼'}</span>
                  {labelOf(c.feature)}
                </li>
              ))}
          </ul>
        </div>
      )}
      <p className="muted small">Model: {result.model_version}</p>
    </section>
  );
}
