import type { PatientFeatures } from './types';

// Feature schema for the patient form. `key` is the exact Z-Alizadeh Sani column name,
// so the backend can map the payload straight onto its training columns.
// LAD, LCX, RCA and Cath are deliberately absent (they are targets / leakage).

export type FeatureKind = 'number' | 'bool' | 'select';
export type GroupId = 'demographics' | 'history' | 'symptoms' | 'exam' | 'ecg' | 'labs' | 'echo';

export interface FeatureDef {
  key: string;
  label: string;
  group: GroupId;
  kind: FeatureKind;
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  /** Typical adult reference range, used for flags in the measurements table. */
  normal?: [number, number];
  /** Computed from other inputs, shown read-only. */
  derived?: boolean;
  /** The model expects a whole number (integer field in the training data). */
  integer?: boolean;
}

export const GROUPS: { id: GroupId; label: string; open: boolean }[] = [
  { id: 'demographics', label: 'Demographics', open: true },
  { id: 'history', label: 'History & risk factors', open: true },
  { id: 'symptoms', label: 'Symptoms', open: true },
  { id: 'exam', label: 'Examination', open: false },
  { id: 'ecg', label: 'ECG', open: true },
  { id: 'labs', label: 'Laboratory', open: false },
  { id: 'echo', label: 'Echocardiography', open: true },
];

const n = (
  key: string, label: string, group: GroupId, unit: string, min: number, max: number, step: number,
  normal?: [number, number], derived?: boolean,
): FeatureDef => ({ key, label, group, kind: 'number', unit, min, max, step, normal, derived });
const b = (key: string, label: string, group: GroupId): FeatureDef => ({ key, label, group, kind: 'bool' });
const s = (key: string, label: string, group: GroupId, options: string[]): FeatureDef => ({ key, label, group, kind: 'select', options });

export const FEATURES: FeatureDef[] = [
  // Demographics
  n('Age', 'Age', 'demographics', 'yr', 18, 100, 1),
  s('Sex', 'Sex', 'demographics', ['Male', 'Female']),
  n('Weight', 'Weight', 'demographics', 'kg', 30, 200, 1),
  n('Length', 'Height', 'demographics', 'cm', 120, 210, 1),
  n('BMI', 'BMI', 'demographics', 'kg/m²', 10, 60, 0.1, [18.5, 25], true),
  // History
  b('DM', 'Diabetes', 'history'),
  b('HTN', 'Hypertension', 'history'),
  b('DLP', 'Dyslipidemia', 'history'),
  b('Current Smoker', 'Current smoker', 'history'),
  b('EX-Smoker', 'Ex-smoker', 'history'),
  b('FH', 'Family history of CAD', 'history'),
  b('Obesity', 'Obesity (BMI > 25)', 'history'),
  b('CRF', 'Chronic renal failure', 'history'),
  b('CVA', 'Prior stroke (CVA)', 'history'),
  b('Airway disease', 'Airway disease', 'history'),
  b('Thyroid Disease', 'Thyroid disease', 'history'),
  b('CHF', 'Heart failure (CHF)', 'history'),
  // Symptoms
  b('Typical Chest Pain', 'Typical chest pain', 'symptoms'),
  b('Atypical', 'Atypical chest pain', 'symptoms'),
  b('Nonanginal', 'Non-anginal pain', 'symptoms'),
  b('Exertional CP', 'Exertional chest pain', 'symptoms'),
  b('LowTH Ang', 'Low-threshold angina', 'symptoms'),
  b('Dyspnea', 'Dyspnea', 'symptoms'),
  n('Function Class', 'Functional class (NYHA)', 'symptoms', '', 0, 4, 1),
  // Exam
  n('BP', 'Blood pressure (systolic)', 'exam', 'mmHg', 70, 240, 1, [90, 130]),
  n('PR', 'Pulse rate', 'exam', 'bpm', 30, 180, 1, [60, 100]),
  b('Edema', 'Edema', 'exam'),
  b('Weak Peripheral Pulse', 'Weak peripheral pulse', 'exam'),
  b('Lung rales', 'Lung rales', 'exam'),
  b('Systolic Murmur', 'Systolic murmur', 'exam'),
  b('Diastolic Murmur', 'Diastolic murmur', 'exam'),
  // ECG
  b('Q Wave', 'Q wave', 'ecg'),
  b('St Elevation', 'ST elevation', 'ecg'),
  b('St Depression', 'ST depression', 'ecg'),
  b('Tinversion', 'T inversion', 'ecg'),
  b('LVH', 'LV hypertrophy', 'ecg'),
  b('Poor R Progression', 'Poor R progression', 'ecg'),
  s('BBB', 'Bundle branch block', 'ecg', ['N', 'LBBB', 'RBBB']),
  // Labs
  n('FBS', 'Fasting blood sugar', 'labs', 'mg/dL', 50, 450, 1, [70, 100]),
  n('CR', 'Creatinine', 'labs', 'mg/dL', 0.3, 6, 0.1, [0.6, 1.3]),
  n('TG', 'Triglycerides', 'labs', 'mg/dL', 30, 1100, 1, [0, 150]),
  n('LDL', 'LDL', 'labs', 'mg/dL', 10, 250, 1, [0, 130]),
  n('HDL', 'HDL', 'labs', 'mg/dL', 10, 120, 1, [40, 100]),
  n('BUN', 'Blood urea nitrogen', 'labs', 'mg/dL', 3, 60, 1, [7, 20]),
  n('ESR', 'ESR', 'labs', 'mm/h', 1, 100, 1, [0, 20]),
  n('HB', 'Hemoglobin', 'labs', 'g/dL', 6, 20, 0.1, [12, 17]),
  n('K', 'Potassium', 'labs', 'mEq/L', 2.5, 7, 0.1, [3.5, 5.0]),
  n('Na', 'Sodium', 'labs', 'mEq/L', 120, 160, 1, [135, 145]),
  n('WBC', 'White blood cells', 'labs', '/µL', 2000, 20000, 100, [4000, 11000]),
  n('Lymph', 'Lymphocytes', 'labs', '%', 5, 70, 1, [20, 40]),
  n('Neut', 'Neutrophils', 'labs', '%', 20, 95, 1, [40, 70]),
  n('PLT', 'Platelets', 'labs', '×10³/µL', 20, 800, 1, [150, 450]),
  // Echo
  n('EF-TTE', 'Ejection fraction (TTE)', 'echo', '%', 10, 80, 1, [55, 70]),
  n('Region RWMA', 'Regions with RWMA', 'echo', '', 0, 4, 1, [0, 0]),
  s('VHD', 'Valvular heart disease', 'echo', ['N', 'mild', 'Moderate', 'Severe']),
];

// Integer fields in the model's input schema (config/model_selection.json); decimals are rejected there.
const INTEGER_KEYS = new Set(['Age', 'BP', 'PR', 'Function Class', 'FBS', 'TG', 'LDL', 'BUN', 'ESR', 'Na', 'WBC', 'Lymph', 'PLT', 'EF-TTE', 'Region RWMA']);
for (const f of FEATURES) if (INTEGER_KEYS.has(f.key)) f.integer = true;

export const FEATURE_BY_KEY: Record<string, FeatureDef> = Object.fromEntries(FEATURES.map((f) => [f.key, f]));

export function labelOf(key: string): string {
  return FEATURE_BY_KEY[key]?.label ?? key;
}

/** Recompute derived fields (BMI, Obesity) after an edit. */
export function withDerived(f: PatientFeatures): PatientFeatures {
  const w = Number(f.Weight);
  const h = Number(f.Length) / 100;
  if (w > 0 && h > 0) {
    const bmi = Math.round((w / (h * h)) * 10) / 10;
    return { ...f, BMI: bmi, Obesity: bmi > 25 ? 1 : 0 };
  }
  return f;
}

const BASE: PatientFeatures = Object.fromEntries(
  FEATURES.map((f) => [f.key, f.kind === 'bool' ? 0 : f.kind === 'select' ? f.options![0] : 0]),
);

const labsNormal: PatientFeatures = {
  FBS: 92, CR: 0.9, TG: 120, LDL: 95, HDL: 48, BUN: 14, ESR: 10, HB: 14, K: 4.2, Na: 140,
  WBC: 7000, Lymph: 32, Neut: 60, PLT: 230,
};

/** A neutral starting point for a new case: healthy adult values, no findings. */
export const NEW_CASE: PatientFeatures = withDerived({
  ...BASE, ...labsNormal, Age: 50, Sex: 'Male', Weight: 72, Length: 172, BP: 120, PR: 72, 'EF-TTE': 60,
});

export interface Preset { id: string; label: string; note: string; features: PatientFeatures }

export const PRESETS: Preset[] = [
  {
    id: 'low',
    label: 'Low risk',
    note: '42 F, non-anginal pain, normal ECG and echo',
    features: withDerived({
      ...BASE, ...labsNormal, Age: 42, Sex: 'Female', Weight: 62, Length: 163, BP: 118, PR: 72,
      Nonanginal: 1, 'EF-TTE': 60, 'Region RWMA': 0,
    }),
  },
  {
    id: 'typical',
    label: 'Typical angina',
    note: '61 M smoker, HTN, typical chest pain, T inversion',
    features: withDerived({
      ...BASE, ...labsNormal, Age: 61, Sex: 'Male', Weight: 82, Length: 172, BP: 142, PR: 78,
      HTN: 1, DLP: 1, 'Current Smoker': 1, 'Typical Chest Pain': 1, 'Exertional CP': 1,
      Tinversion: 1, 'EF-TTE': 48, 'Region RWMA': 1, LDL: 148, HDL: 36, TG: 190,
    }),
  },
  {
    id: 'multi',
    label: 'Multi-vessel',
    note: '70 M diabetic, ST depression, Q waves, EF 32%',
    features: withDerived({
      ...BASE, ...labsNormal, Age: 70, Sex: 'Male', Weight: 88, Length: 170, BP: 156, PR: 64,
      DM: 1, HTN: 1, DLP: 1, FH: 1, 'EX-Smoker': 1, 'Typical Chest Pain': 1, Dyspnea: 1,
      'Function Class': 2, 'Q Wave': 1, 'St Elevation': 1, 'St Depression': 1, Tinversion: 1,
      'Poor R Progression': 1, FBS: 182, CR: 1.4, LDL: 160, HDL: 31, TG: 240, ESR: 34,
      'EF-TTE': 32, 'Region RWMA': 3,
    }),
  },
];
