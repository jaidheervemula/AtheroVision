// Clinical sanity checks shown alongside predictions:
//  1. rhythmOf: names the rhythm the schematic ECG is showing.
//  2. clinicalFlags: values that are dangerous or physiologically extreme.
//  3. outOfRange: inputs outside what the model saw in training, where predictions are less reliable.
// These are display aids for a demo, not diagnostic rules.

import { FEATURE_BY_KEY, labelOf } from './features';
import type { PatientFeatures } from './types';

export function rhythmOf(bpm: number): { label: string; level: 'ok' | 'warn' } {
  if (bpm < 60) return { label: 'Sinus bradycardia', level: 'warn' };
  if (bpm > 100) return { label: 'Sinus tachycardia', level: 'warn' };
  return { label: 'Normal sinus rhythm', level: 'ok' };
}

export interface Flag { key: string; level: 'warn' | 'danger'; text: string }

type Rule = (v: number) => Omit<Flag, 'key'> | null;
const RULES: Record<string, Rule> = {
  BMI: (v) => v < 16 ? { level: 'danger', text: `BMI ${v}, severely underweight` }
    : v < 18.5 ? { level: 'warn', text: `BMI ${v}, underweight` }
    : v >= 40 ? { level: 'warn', text: `BMI ${v}, severe obesity` } : null,
  PR: (v) => v < 50 ? { level: 'warn', text: `Pulse ${v}, marked bradycardia` }
    : v > 120 ? { level: 'warn', text: `Pulse ${v}, marked tachycardia` } : null,
  BP: (v) => v < 90 ? { level: 'danger', text: `Systolic BP ${v}, hypotension` }
    : v >= 180 ? { level: 'danger', text: `Systolic BP ${v}, hypertensive crisis range` } : null,
  HDL: (v) => v < 25 ? { level: 'warn', text: `HDL ${v}, very low` } : null,
  LDL: (v) => v >= 190 ? { level: 'warn', text: `LDL ${v}, very high` } : null,
  FBS: (v) => v < 70 ? { level: 'danger', text: `Glucose ${v}, hypoglycaemia` }
    : v >= 200 ? { level: 'warn', text: `Glucose ${v}, marked hyperglycaemia` } : null,
  'EF-TTE': (v) => v < 35 ? { level: 'danger', text: `EF ${v}%, severely reduced` } : null,
  K: (v) => v < 3.0 || v > 5.5 ? { level: 'danger', text: `Potassium ${v}, dangerous level` } : null,
  Na: (v) => v < 130 || v > 150 ? { level: 'warn', text: `Sodium ${v}, abnormal` } : null,
  HB: (v) => v < 8 ? { level: 'danger', text: `Hemoglobin ${v}, severe anaemia` } : null,
};

export function clinicalFlags(f: PatientFeatures): Flag[] {
  const out: Flag[] = [];
  for (const [key, rule] of Object.entries(RULES)) {
    const v = Number(f[key]);
    if (!Number.isFinite(v)) continue;
    const r = rule(v);
    if (r) out.push({ key, ...r });
  }
  return out.sort((a, b) => (a.level === b.level ? 0 : a.level === 'danger' ? -1 : 1));
}

// Min/max of each numeric model input in the training data (303 patients), copied from the ML
// project's config/model_selection.json -> input_schema. When the model service is connected,
// its /metrics feature_ranges are used instead (same source).
export const TRAINING_RANGES: Record<string, [number, number]> = {
  Age: [30, 86], BMI: [18.12, 40.9], BP: [90, 190], PR: [50, 110], 'Function Class': [0, 3], FBS: [62, 400], CR: [0.5, 2.2], TG: [37, 1050], LDL: [18, 232], HDL: [15.9, 111.0], BUN: [6, 52], ESR: [1, 90], HB: [8.9, 17.6], K: [3.0, 6.6], Na: [128, 156], WBC: [3700, 18000], Lymph: [7, 60], PLT: [25, 742], 'EF-TTE': [15, 60], 'Region RWMA': [0, 4],
};

export interface OutOfRange { key: string; label: string; value: number; range: [number, number]; unit: string }

export function outOfRange(f: PatientFeatures, ranges: Record<string, [number, number]> = TRAINING_RANGES): OutOfRange[] {
  const out: OutOfRange[] = [];
  for (const [key, range] of Object.entries(ranges)) {
    const v = Number(f[key]);
    if (!Number.isFinite(v)) continue;
    if (v < range[0] || v > range[1]) {
      out.push({ key, label: labelOf(key), value: v, range, unit: FEATURE_BY_KEY[key]?.unit ?? '' });
    }
  }
  return out;
}
