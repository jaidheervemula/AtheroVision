// Single place the UI talks to the model. With VITE_API_URL empty it uses the labelled demo model.
// `npm run dev` uses .env (the local model service); `npm run dev:demo` uses .env.demo (no service).
import { mockPredict } from './mockModel';
import { TARGETS, type DatasetPatient, type MetricsResponse, type PatientFeatures, type PredictionResponse } from '../types';

export const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '');

export const apiMode: 'mock' | 'api' = API_URL ? 'api' : 'mock';

export class PredictionError extends Error {
  readonly errors: string[];
  constructor(message: string, errors: string[] = []) {
    super(message);
    this.errors = errors;
  }
}

interface PredictOptions {
  signal?: AbortSignal;
  /** Include per-feature explanations. The real LAD explanation takes about 2 s, so the
   *  interface asks for probabilities first and explanations second. */
  explain?: boolean;
  /** The row when these are an unedited dataset patient (left out of their own reality check). */
  datasetRow?: number | null;
}

export async function predict(features: PatientFeatures, { signal, explain = true, datasetRow = null }: PredictOptions = {}): Promise<PredictionResponse> {
  if (!API_URL) return mockPredict(features);
  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(datasetRow == null ? { features, explain } : { features, explain, dataset_row: datasetRow }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new PredictionError(`Can't reach the model service at ${API_URL}. Start it from the ML project folder with: python -m uvicorn api.main:app --port 8000`);
  }
  if (!res.ok) {
    let errors: string[] = [];
    try {
      const body = await res.json();
      errors = body?.detail?.errors ?? [];
    } catch { /* not JSON */ }
    throw new PredictionError(
      res.status === 422 && errors.length ? `The model rejected the input: ${errors.join(' ')}` : `The model service returned an error (${res.status}).`,
      errors,
    );
  }
  const data = (await res.json()) as Omit<PredictionResponse, 'source'>;
  // A probabilities-only response has no explanation fields; give them safe empty values.
  for (const t of TARGETS) {
    const p = data.predictions[t];
    p.contributions ??= [];
    p.base_value ??= 0;
  }
  return { ...data, source: 'api' };
}

export async function fetchMetrics(): Promise<MetricsResponse | null> {
  if (!API_URL) return null;
  try {
    const res = await fetch(`${API_URL}/metrics`);
    return res.ok ? ((await res.json()) as MetricsResponse) : null;
  } catch {
    return null;
  }
}

/** A real training patient with its angiography outcome and cross-validated prediction. */
export async function fetchDatasetPatient(row: number): Promise<DatasetPatient> {
  if (!API_URL) throw new PredictionError('Dataset patients need the model service (npm run dev).');
  let res: Response;
  try {
    res = await fetch(`${API_URL}/dataset/${row}`);
  } catch {
    throw new PredictionError(`Can't reach the model service at ${API_URL}.`);
  }
  if (res.status === 404) throw new PredictionError('That row is not in the dataset.');
  if (!res.ok) throw new PredictionError(`The model service returned an error (${res.status}).`);
  return (await res.json()) as DatasetPatient;
}
