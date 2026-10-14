// Shared types for the prediction contract between the frontend and the Python API.
// Keep this in sync with README.md → "API contract".

export const TARGETS = ['CAD', 'LAD', 'LCX', 'RCA'] as const;
export type Target = (typeof TARGETS)[number];

export const VESSELS = ['LAD', 'LCX', 'RCA'] as const;
export type Vessel = (typeof VESSELS)[number];

/** Booleans are sent as 0/1, categoricals as their option string. */
export type FeatureValue = number | string;
export type PatientFeatures = Record<string, FeatureValue>;

export interface Contribution {
  /** Exact dataset column name, e.g. "Typical Chest Pain". */
  feature: string;
  /** The value the model saw for this patient. */
  value: FeatureValue;
  /** Contribution to the model output, in the target's `units` (positive = pushes risk up). */
  shap: number;
}

/** log-odds: exact additive decomposition (logistic regression). probability: approximate impact (random forest). */
export type ExplanationUnits = 'log-odds' | 'probability';

export interface TargetPrediction {
  probability: number; // 0..1
  label: 0 | 1;
  threshold: number; // decision threshold used for `label`
  /** Reference value in the same units as the contributions. */
  base_value: number;
  contributions: Contribution[];
  units?: ExplanationUnits; // missing = log-odds
  additive?: boolean;
  /** The reference expressed as a probability (shown as "reference patient"). */
  reference_probability?: number;
  method?: string;
  /** Inputs this target's model does not use at all. */
  features_not_used?: string[];
  model_name?: string;
  model_type?: string;
  n_input_features?: number;
  /** The package's own 0.5 decision, kept for reference when `threshold` is a screening cut-off. */
  default_threshold?: number;
  default_label?: 0 | 1;
  /** Why this target is flagged at `threshold` (api/operating_points.py). */
  operating_point?: { mode: 'default' | 'screening'; reason: string; cv_sensitivity?: number | null; cv_specificity?: number | null };
  /** Observed outcome among the dataset patients whose cross-validated score was closest to this one. */
  reality_check?: RealityCheck;
}

export interface RealityCheck {
  n: number;
  positives: number;
  rate: number;
  ci_low: number;
  ci_high: number;
  score_min: number;
  score_max: number;
  /** Set when the loaded dataset patient was left out of their own comparison. */
  excluded_row?: number | null;
}

export interface PredictionResponse {
  model_version: string;
  source: 'mock' | 'api';
  predictions: Record<Target, TargetPrediction>;
  /** False when only probabilities were requested; explanations then come from an earlier result. */
  explained?: boolean;
  /** The CAD model and the vessel models are independent and may disagree; never adjusted to agree. */
  agreement?: { models_disagree: boolean; note: string };
  warnings?: string[];
  /** Form fields none of the models use. */
  unused_inputs?: string[];
  disclaimer?: string;
  limitations?: string[];
  threshold_note?: string;
}

export interface TargetMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc?: number;
  brier?: number;
  balanced_accuracy?: number;
  specificity?: number;
}

export interface MetricsResponse {
  validation: string; // e.g. "Repeated stratified 5-fold CV, 5 repeats"
  n_samples: number;
  targets: Record<Target, TargetMetrics>;
  /** Standard deviation of each metric across CV folds. */
  sd?: Record<Target, Partial<TargetMetrics>>;
  models?: Record<Target, { name: string; type: string; n_input_features: number }>;
  limitations?: string[];
  /** Optional exact [min, max] of each numeric training feature, for out-of-range warnings. */
  feature_ranges?: Record<string, [number, number]>;
  /** Flagging cut-off per target with cross-validated metrics there and at 0.5. */
  operating_points?: Partial<Record<Target, {
    threshold: number; mode: 'default' | 'screening'; reason: string;
    cv: { sensitivity: number; specificity: number } | null;
    cv_at_default: { sensitivity: number; specificity: number } | null;
  }>>;
  operating_rule?: string;
  /** Predicted vs observed from the saved cross-validation reports. */
  calibration?: Partial<Record<Target, CalibrationInfo>>;
  calibration_note?: string;
}

export interface CalibrationInfo {
  bins: { n: number; mean_predicted: number; observed: number; ci_low: number; ci_high: number }[];
  brier?: { estimate: number; ci_low: number; ci_high: number };
  calibration_slope?: { estimate: number; ci_low: number; ci_high: number };
}

/** A real patient from the training dataset (GET /dataset/{row}). */
export interface DatasetPatient {
  row: number;
  n_rows: number;
  features: PatientFeatures;
  /** Angiography outcome, 1 = disease/stenosis. For comparison only; never sent to the model. */
  actual: Record<Target, 0 | 1>;
  actual_source: string;
  /** Mean out-of-fold probability: the prediction made while this patient was held out of training. */
  cv: Partial<Record<Target, { probability: number; min: number; max: number; repeats: number }>>;
  cv_source: string;
  note: string;
  /** Set when the outcome labels contradict each other (CAD = no, but a vessel stenotic). */
  label_conflict?: string | null;
}
