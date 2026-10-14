import { labelOf } from './features';
import { pct, riskBand } from './risk';
import { VESSELS, type PredictionResponse, type Target } from './types';

/** "T inversion" -> "T inversion", "Typical chest pain" -> "typical chest pain", "LDL" stays "LDL". */
function prose(label: string) {
  return /^[A-Z][a-z]/.test(label) ? label[0].toLowerCase() + label.slice(1) : label;
}

function list(items: string[]) {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

export function topDrivers(r: PredictionResponse, t: Target, n = 3, direction: 'up' | 'down' = 'up') {
  return [...r.predictions[t].contributions]
    .filter((c) => (direction === 'up' ? c.shap > 0.01 : c.shap < -0.01))
    .sort((a, b) => (direction === 'up' ? b.shap - a.shap : a.shap - b.shap))
    .slice(0, n);
}

/** Plain-language reading of the prediction, in the order a clinician reads a report. */
export function assessmentSummary(r: PredictionResponse): string[] {
  const cad = r.predictions.CAD.probability;
  const band = riskBand(cad).toLowerCase();
  const vessels = VESSELS.map((v) => ({ v, p: r.predictions[v].probability })).sort((a, b) => b.p - a.p);
  const flagged = vessels.filter((x) => x.p >= r.predictions[x.v].threshold);
  const out = [`${band[0].toUpperCase()}${band.slice(1)} probability of coronary artery disease (${pct(cad)}).`];
  if (flagged.length) {
    out.push(`Stenosis flagged in the ${list(flagged.map((x) => `${x.v} (${pct(x.p)})`))}.`);
  } else {
    out.push(`No vessel reaches its flagging cut-off; the highest is the ${vessels[0].v} (${pct(vessels[0].p)}).`);
  }
  const drivers = topDrivers(r, 'CAD', 3).map((c) => prose(labelOf(c.feature)));
  if (drivers.length) out.push(`Main contributors: ${list(drivers)}.`);
  return out;
}
