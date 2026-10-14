// MOCK predictor so the UI can be built before the real models are served.
// It is a hand-weighted logistic model, NOT trained and NOT clinically meaningful.
// Contributions are exact SHAP values for a linear model: phi_i = w_i * (x_i - E[x_i]),
// so they sum (with base_value) to the predicted log-odds, just like TreeSHAP output will.

import { FEATURES } from '../features';
import type { PatientFeatures, PredictionResponse, Target, TargetPrediction } from '../types';

// Approximate population statistics in the shape of the Z-Alizadeh Sani cohort.
const NUM_STATS: Record<string, [mean: number, sd: number]> = {
  Age: [58.9, 10.4], Weight: [73.8, 12], Length: [164.7, 9.3], BMI: [27.2, 4.1], BP: [129.6, 18.9],
  PR: [75.1, 8.9], FBS: [119, 52], CR: [1.06, 0.26], TG: [150, 97], LDL: [104, 35], HDL: [40, 10.5],
  BUN: [17.5, 7], ESR: [19.5, 15.9], HB: [13.2, 1.6], K: [4.23, 0.46], Na: [141, 3.8], WBC: [7562, 2413],
  Lymph: [32.4, 9.9], Neut: [60.1, 10.2], PLT: [221, 60.8], 'EF-TTE': [47.2, 8.9], 'Region RWMA': [0.62, 1.13],
  'Function Class': [0.66, 1.03],
};
const PREVALENCE: Record<string, number> = {
  DM: 0.3, HTN: 0.59, 'Current Smoker': 0.21, 'EX-Smoker': 0.03, FH: 0.16, Obesity: 0.83, CRF: 0.02,
  CVA: 0.02, 'Airway disease': 0.04, 'Thyroid Disease': 0.02, CHF: 0.01, DLP: 0.37, Edema: 0.04,
  'Weak Peripheral Pulse': 0.02, 'Lung rales': 0.03, 'Systolic Murmur': 0.14, 'Diastolic Murmur': 0.03,
  'Typical Chest Pain': 0.54, Dyspnea: 0.44, Atypical: 0.27, Nonanginal: 0.05, 'Exertional CP': 0.01,
  'LowTH Ang': 0.01, 'Q Wave': 0.05, 'St Elevation': 0.05, 'St Depression': 0.23, Tinversion: 0.3,
  LVH: 0.02, 'Poor R Progression': 0.03,
};
const OPTION_FREQ: Record<string, Record<string, number>> = {
  Sex: { Male: 0.58, Female: 0.42 },
  BBB: { N: 0.93, LBBB: 0.04, RBBB: 0.03 },
  VHD: { N: 0.39, mild: 0.48, Moderate: 0.1, Severe: 0.03 },
};

// Weights: per SD for numeric, per presence for boolean, per option for categoricals.
type Weights = Record<string, number | Record<string, number>>;
const CAD_W: Weights = {
  Age: 0.75, Sex: { Male: 0.35, Female: 0 }, DM: 0.7, HTN: 0.5, 'Current Smoker': 0.45, 'EX-Smoker': 0.2,
  FH: 0.35, DLP: 0.3, BMI: 0.05, Obesity: 0.05, BP: 0.2, PR: 0.1, 'Typical Chest Pain': 1.4, Atypical: -0.9,
  Nonanginal: -1.1, Dyspnea: 0.15, 'Function Class': 0.15, 'Exertional CP': 0.3, 'LowTH Ang': 0.3,
  'Q Wave': 0.6, 'St Elevation': 0.5, 'St Depression': 0.6, Tinversion: 0.8, LVH: 0.2,
  'Poor R Progression': 0.3, BBB: { N: 0, LBBB: 0.4, RBBB: 0.15 }, FBS: 0.35, CR: 0.1, TG: 0.2, LDL: 0.15,
  HDL: -0.25, BUN: 0.05, ESR: 0.2, HB: -0.05, K: 0.05, WBC: 0.05, Lymph: -0.05, Neut: 0.05,
  'EF-TTE': -0.55, 'Region RWMA': 0.6, VHD: { N: 0, mild: -0.1, Moderate: -0.3, Severe: -0.5 },
  CRF: 0.2, CVA: 0.2, CHF: 0.2, Edema: 0.05, 'Weak Peripheral Pulse': 0.2, 'Lung rales': 0.1,
  'Systolic Murmur': -0.1, 'Diastolic Murmur': -0.1,
};
const VESSEL_DELTA: Record<'LAD' | 'LCX' | 'RCA', Weights> = {
  LAD: { Tinversion: 0.3, 'Q Wave': 0.3, 'EF-TTE': -0.2, 'Region RWMA': 0.1, 'Poor R Progression': 0.3, 'Typical Chest Pain': 0.1 },
  LCX: { 'St Depression': 0.3, DM: 0.2, Age: 0.1, LVH: 0.2 },
  RCA: { 'St Elevation': 0.3, HTN: 0.2, 'Current Smoker': 0.2, BP: 0.15, PR: -0.15, 'Q Wave': 0.1 },
};
// Reference (average-patient) log-odds ≈ cohort prevalence of each label.
const BASE_LOGIT: Record<Target, number> = { CAD: 0.9, LAD: 0.15, LCX: -0.9, RCA: -0.65 };

// Shrink the shared weights so probabilities stay in a plausible range (the raw weights
// stack up quickly across ~55 features). Vessel models share a weaker copy of the CAD signal.
const SHARED_SCALE: Record<Target, number> = { CAD: 0.5, LAD: 0.38, LCX: 0.3, RCA: 0.32 };

function weightsFor(t: Target): Weights {
  const k0 = SHARED_SCALE[t];
  const w: Weights = {};
  for (const [k, v] of Object.entries(CAD_W)) {
    w[k] = typeof v === 'number' ? v * k0 : Object.fromEntries(Object.entries(v).map(([o, x]) => [o, x * k0]));
  }
  if (t === 'CAD') return w;
  for (const [k, v] of Object.entries(VESSEL_DELTA[t])) w[k] = ((w[k] as number) ?? 0) + (v as number);
  return w;
}

/** Centred contribution of one feature: w * (x - E[x]). */
function phi(key: string, value: number | string, w: number | Record<string, number>): number {
  if (typeof w === 'number') {
    const stats = NUM_STATS[key];
    if (stats) return w * ((Number(value) - stats[0]) / stats[1]);
    return w * (Number(value) - (PREVALENCE[key] ?? 0));
  }
  const freq = OPTION_FREQ[key] ?? {};
  const expected = Object.entries(freq).reduce((acc, [o, p]) => acc + p * (w[o] ?? 0), 0);
  return (w[String(value)] ?? 0) - expected;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

function predictTarget(t: Target, f: PatientFeatures): TargetPrediction {
  const W = weightsFor(t);
  const contributions = FEATURES.map((def) => ({
    feature: def.key,
    value: f[def.key],
    shap: W[def.key] === undefined ? 0 : phi(def.key, f[def.key], W[def.key]),
  }));
  const logit = BASE_LOGIT[t] + contributions.reduce((a, c) => a + c.shap, 0);
  const probability = sigmoid(logit);
  return {
    probability, label: probability >= 0.5 ? 1 : 0, threshold: 0.5, base_value: BASE_LOGIT[t], contributions,
    units: 'log-odds', additive: true, reference_probability: sigmoid(BASE_LOGIT[t]),
    model_name: 'demo (hand-weighted)', n_input_features: FEATURES.length,
  };
}

export function mockPredict(f: PatientFeatures): PredictionResponse {
  return {
    model_version: 'mock-0.1 (hand-weighted, untrained)',
    source: 'mock',
    explained: true,
    predictions: {
      CAD: predictTarget('CAD', f),
      LAD: predictTarget('LAD', f),
      LCX: predictTarget('LCX', f),
      RCA: predictTarget('RCA', f),
    },
  };
}
