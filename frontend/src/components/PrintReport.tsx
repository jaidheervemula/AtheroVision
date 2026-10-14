import { FEATURES, GROUPS } from '../features';
import { TARGETS, type PatientFeatures, type PredictionResponse } from '../types';
import { VESSEL_INFO, pct, riskBand } from '../risk';
import { assessmentSummary, topDrivers } from '../summary';
import { clinicalFlags, outOfRange } from '../clinical';
import { labelOf } from '../features';
import { formatValue } from './format';
import { DEMO_NOTE, DISCLAIMER } from '../disclaimer';

// Printable clinical report. Hidden on screen, shown only when printing (Export report or Ctrl+P),
// so the browser's "Save as PDF" produces a clean, light, A4 document.

interface Props {
  features: PatientFeatures;
  result: PredictionResponse | null;
  caseId: string;
  caseLabel: string;
  image: string | null;
  generatedAt: Date;
  ranges?: Record<string, [number, number]>;
}

export default function PrintReport({ features, result, caseId, caseLabel, image, generatedAt, ranges }: Props) {
  if (!result) return null;
  const flags = clinicalFlags(features);
  const ood = outOfRange(features, ranges);

  return (
    <article className="report" aria-hidden>
      <header className="report__head">
        <div>
          <h1>Coronary risk assessment</h1>
          <p>AtheroVision report for {caseLabel}</p>
        </div>
        <table className="report__meta">
          <tbody>
            <tr><th>Case</th><td>{caseId}</td></tr>
            <tr><th>Generated</th><td>{generatedAt.toLocaleString()}</td></tr>
            <tr><th>Model</th><td>{result.model_version}</td></tr>
          </tbody>
        </table>
      </header>

      <p className="report__disclaimer">
        {DISCLAIMER}
        {result.source === 'mock' && <><br /><b>{DEMO_NOTE}</b></>}
      </p>

      <section className="report__section">
        <h2>Assessment</h2>
        <p className="report__summary">{assessmentSummary(result).join(' ')}</p>
        <div className="report__grid">
          {image && <img className="report__img" src={image} alt="3D heart with coronary arteries coloured by predicted stenosis risk" />}
          <table className="report__table">
            <thead><tr><th>Target</th><th>Probability</th><th>Result</th><th>Cut-off</th><th>Band</th><th title="Dataset patients with the closest cross-validated score who actually had the condition">Similar patients</th></tr></thead>
            <tbody>
              {TARGETS.map((t) => {
                const pr = result.predictions[t];
                return (
                  <tr key={t}>
                    <td><b>{t}</b>{t !== 'CAD' && <span className="report__sub">{VESSEL_INFO[t as keyof typeof VESSEL_INFO].name}</span>}</td>
                    <td className="num">{pct(pr.probability)}</td>
                    <td>{t === 'CAD' ? (pr.label ? 'CAD predicted' : 'No CAD predicted') : pr.label ? 'Stenosis ≥50% flagged' : 'Not flagged'}</td>
                    <td className="num">{pct(pr.threshold)}{pr.operating_point?.mode === 'screening' ? ' (screening)' : ''}</td>
                    <td>{riskBand(pr.probability)}</td>
                    <td className="num">{pr.reality_check ? `${pr.reality_check.positives}/${pr.reality_check.n} had it` : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="report__section">
        <h2>Why the model predicted this</h2>
        <table className="report__table report__table--compact">
          <thead><tr><th>Target</th><th>Raises risk</th><th>Lowers risk</th></tr></thead>
          <tbody>
            {TARGETS.map((t) => (
              <tr key={t}>
                <td><b>{t}</b></td>
                <td>{topDrivers(result, t, t === 'CAD' ? 5 : 3, 'up').map((c) => `${labelOf(c.feature)} (${formatValue(c.feature, c.value)})`).join('; ') || '—'}</td>
                <td>{topDrivers(result, t, 3, 'down').map((c) => `${labelOf(c.feature)} (${formatValue(c.feature, c.value)})`).join('; ') || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="report__note">
          {TARGETS.some((t) => result.predictions[t].units === 'probability')
            ? `Exact SHAP values (log-odds, relative to a reference patient) for ${TARGETS.filter((t) => result.predictions[t].units !== 'probability').join(', ')}; approximate single-input impacts, not SHAP, for ${TARGETS.filter((t) => result.predictions[t].units === 'probability').join(', ')} (random forest, not additive).`
            : 'Each input\u2019s contribution to the model\u2019s log-odds, relative to a reference patient.'}
          {' '}Contributions are statistical associations from the training data, not causes.
        </p>
        {result.agreement?.models_disagree && (
          <p className="report__note"><b>Models disagree:</b> {result.agreement.note}</p>
        )}
      </section>

      {(flags.length > 0 || ood.length > 0) && (
        <section className="report__section">
          <h2>Alerts</h2>
          <ul className="report__list">
            {flags.map((f) => <li key={f.key}><b>{f.level === 'danger' ? 'Critical' : 'Warning'}:</b> {f.text}</li>)}
            {ood.map((o) => <li key={o.key}><b>Outside training data:</b> {o.label} {o.value} {o.unit} (trained on {o.range[0]}–{o.range[1]}); prediction less reliable.</li>)}
          </ul>
        </section>
      )}

      <section className="report__section">
        <h2>Patient inputs</h2>
        <div className="report__inputs">
          {GROUPS.map((g) => (
            <table key={g.id} className="report__table report__table--compact">
              <thead><tr><th colSpan={2}>{g.label}</th></tr></thead>
              <tbody>
                {FEATURES.filter((d) => d.group === g.id).map((d) => (
                  <tr key={d.key}><td>{d.label}</td><td className="num">{formatValue(d.key, features[d.key])}</td></tr>
                ))}
              </tbody>
            </table>
          ))}
        </div>
      </section>

      <footer className="report__foot">
        Vessel-level predictions: colour on the 3D model does not indicate where along a vessel a lesion is.
        Anatomy: BodyParts3D, © DBCLS, CC BY-SA 2.1 JP.
      </footer>
    </article>
  );
}
