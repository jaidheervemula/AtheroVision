// Contract tests between the frontend and the model service (api/ in the ML project).
// Run with `npm test`. They need no running service: fetch is replaced with a stub.

import { afterEach, describe, expect, it, vi } from 'vitest';
import schemaFile from './__fixtures__/model_input_schema.json';
import { FEATURES, NEW_CASE, PRESETS } from '../features';
import { API_URL, PredictionError, apiMode, fetchDatasetPatient, predict } from './client';
import { TARGETS } from '../types';

type Spec =
  | { kind: 'numeric'; dtype: 'integer' | 'float'; training_min: number; training_max: number }
  | { kind: 'categorical'; allowed_values: (string | number)[] };
const schema = schemaFile.input_schema as Record<string, Spec>;
const byKey = Object.fromEntries(FEATURES.map((f) => [f.key, f]));

// How the service re-encodes frontend values (api/adapter.py); mirrored here to check every option.
const SEX_TO_MODEL: Record<string, string> = { Female: 'Fmale', Male: 'Male' };

describe('form fields match the model input schema', () => {
  it('collects every field the models need, under the same name', () => {
    const missing = Object.keys(schema).filter((k) => !byKey[k]);
    expect(missing).toEqual([]);
  });

  it('never collects a prediction target', () => {
    for (const t of schemaFile.targets) expect(byKey[t]).toBeUndefined();
  });

  it('uses a compatible control and value for every model field', () => {
    for (const [key, spec] of Object.entries(schema)) {
      const def = byKey[key];
      if (spec.kind === 'numeric') {
        expect(def.kind, key).toBe('number');
        expect(!!def.integer, `${key} integer`).toBe(spec.dtype === 'integer');
        continue;
      }
      const allowed = spec.allowed_values.map(String);
      if (key === 'Sex') {
        for (const o of def.options!) expect(allowed, o).toContain(SEX_TO_MODEL[o]);
      } else if (allowed.join() === '0,1' || allowed.join() === 'N,Y') {
        expect(def.kind, key).toBe('bool');           // sent as 0/1, re-encoded to N/Y where needed
      } else {
        expect(def.kind, key).toBe('select');
        for (const o of def.options!) expect(allowed, `${key}=${o}`).toContain(o);
      }
    }
  });

  it('gives every example patient a complete, well-typed value for every model field', () => {
    for (const f of [...PRESETS.map((p) => p.features), NEW_CASE]) {
      for (const [key, spec] of Object.entries(schema)) {
        const v = f[key];
        expect(v, key).toBeDefined();
        if (spec.kind === 'numeric') {
          expect(typeof v, key).toBe('number');
          if (spec.dtype === 'integer') expect(Number.isInteger(v), `${key}=${v}`).toBe(true);
        }
      }
    }
  });
});

describe('prediction client', () => {
  afterEach(() => vi.unstubAllGlobals());

  const ok = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));
  const probsOnly = {
    model_version: 'test', explained: false,
    predictions: Object.fromEntries(TARGETS.map((t) => [t, { probability: 0.4, label: 0, threshold: 0.5 }])),
    agreement: { models_disagree: false, note: '' }, warnings: [], unused_inputs: [],
  };

  it('runs in real-API mode when VITE_API_URL is set (.env)', () => {
    expect(API_URL).toBe('http://localhost:8000');
    expect(apiMode).toBe('api');
  });

  it('posts the complete feature object and the explain flag', async () => {
    const fetchMock = vi.fn(() => ok(probsOnly));
    vi.stubGlobal('fetch', fetchMock);
    const features = PRESETS[1].features;
    await predict(features, { explain: false });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('http://localhost:8000/predict');
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');
    const body = JSON.parse(init.body as string);
    expect(body.explain).toBe(false);
    expect(Object.keys(body.features).sort()).toEqual(Object.keys(features).sort());
    for (const t of schemaFile.targets) expect(body.features[t]).toBeUndefined();
  });

  it('sends the dataset row only for an unedited dataset patient', async () => {
    const fetchMock = vi.fn(() => ok(probsOnly));
    vi.stubGlobal('fetch', fetchMock);
    await predict(NEW_CASE, { explain: false, datasetRow: 32 });
    await predict(NEW_CASE, { explain: false });
    const bodies = (fetchMock.mock.calls as unknown as [string, RequestInit][]).map(([, init]) => JSON.parse(init.body as string));
    expect(bodies[0].dataset_row).toBe(32);
    expect('dataset_row' in bodies[1]).toBe(false);
  });

  it('labels the result as coming from the API and fills empty explanation fields', async () => {
    vi.stubGlobal('fetch', vi.fn(() => ok(probsOnly)));
    const r = await predict(NEW_CASE, { explain: false });
    expect(r.source).toBe('api');
    for (const t of TARGETS) {
      expect(r.predictions[t].contributions).toEqual([]);
      expect(r.predictions[t].base_value).toBe(0);
    }
  });

  it('reports every validation problem from a 422 response', async () => {
    const detail = { message: 'Invalid patient input', errors: ["'Age' must be a whole number, got 61.5.", "Missing field 'BP'."] };
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(JSON.stringify({ detail }), { status: 422 }))));
    const err = await predict(NEW_CASE).catch((e) => e);
    expect(err).toBeInstanceOf(PredictionError);
    expect(err.errors).toEqual(detail.errors);
    expect(err.message).toContain('whole number');
  });

  it('reports a server error without inventing a result', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response('oops', { status: 500 }))));
    await expect(predict(NEW_CASE)).rejects.toThrow('returned an error (500)');
  });

  it('never falls back to demo predictions when the service is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))));
    await expect(predict(NEW_CASE)).rejects.toThrow("Can't reach the model service at http://localhost:8000");
  });

  it('passes cancellation through unchanged', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new DOMException('aborted', 'AbortError'))));
    await expect(predict(NEW_CASE)).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('loads a dataset patient from /dataset/{row} and reports a missing row', async () => {
    const body = { row: 5, n_rows: 303, features: { Age: 60 }, actual: { CAD: 1, LAD: 0, LCX: 0, RCA: 1 }, cv: {}, note: '' };
    const fetchMock = vi.fn(() => ok(body));
    vi.stubGlobal('fetch', fetchMock);
    const d = await fetchDatasetPatient(5);
    expect((fetchMock.mock.calls[0] as unknown as [string])[0]).toBe('http://localhost:8000/dataset/5');
    expect(d.actual.CAD).toBe(1);
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response('{}', { status: 404 }))));
    await expect(fetchDatasetPatient(999)).rejects.toThrow('not in the dataset');
  });
});
