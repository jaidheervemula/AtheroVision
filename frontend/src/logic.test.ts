// Unit tests for the logic the UI depends on. Run with `npm test`.
// These run against the demo model today; once the real API is connected the contract tests
// (shape of a prediction) still apply, and the mock-specific ones document how the demo behaves.

import { describe, expect, it } from 'vitest';
import { mockPredict } from './api/mockModel';
import { FEATURES, NEW_CASE, PRESETS, withDerived } from './features';
import { clinicalFlags, outOfRange, rhythmOf, TRAINING_RANGES } from './clinical';
import { assessmentSummary, topDrivers } from './summary';
import { pct, riskBand, riskColor, sigmoid } from './risk';
import { TARGETS } from './types';

const typical = PRESETS.find((p) => p.id === 'typical')!.features;

describe('feature schema', () => {
  it('never includes the target or leakage columns as inputs', () => {
    const keys = FEATURES.map((f) => f.key);
    for (const banned of ['LAD', 'LCX', 'RCA', 'Cath']) expect(keys).not.toContain(banned);
  });

  it('gives every feature a value in each preset and the new-case template', () => {
    for (const f of [...PRESETS.map((p) => p.features), NEW_CASE]) {
      for (const def of FEATURES) expect(f[def.key], def.key).toBeDefined();
    }
  });

  it('derives BMI and obesity from weight and height', () => {
    const f = withDerived({ ...NEW_CASE, Weight: 90, Length: 170 });
    expect(f.BMI).toBeCloseTo(31.1, 1);
    expect(f.Obesity).toBe(1);
    expect(withDerived({ ...NEW_CASE, Weight: 60, Length: 175 }).Obesity).toBe(0);
  });
});

describe('prediction contract', () => {
  const r = mockPredict(typical);

  it('returns all four targets with valid probabilities and labels', () => {
    for (const t of TARGETS) {
      const p = r.predictions[t];
      expect(p.probability).toBeGreaterThanOrEqual(0);
      expect(p.probability).toBeLessThanOrEqual(1);
      expect(p.label).toBe(p.probability >= p.threshold ? 1 : 0);
      expect(p.contributions.length).toBe(FEATURES.length);
    }
  });

  it('has SHAP values that add up to the prediction (base value + sum = log-odds)', () => {
    for (const t of TARGETS) {
      const p = r.predictions[t];
      const logit = p.base_value + p.contributions.reduce((a, c) => a + c.shap, 0);
      expect(sigmoid(logit)).toBeCloseTo(p.probability, 9);
    }
  });

  it('ranks the example patients in the expected order of risk', () => {
    const cad = (id: string) => mockPredict(PRESETS.find((p) => p.id === id)!.features).predictions.CAD.probability;
    expect(cad('low')).toBeLessThan(cad('typical'));
    expect(cad('typical')).toBeLessThan(cad('multi'));
  });
});

describe('risk display', () => {
  it('maps probabilities onto the colour scale end to end', () => {
    expect(riskColor(0)).toBe('#5fe0a0');
    expect(riskColor(1)).toBe('#ff5a6e');
    expect(riskColor(-1)).toBe(riskColor(0));
    expect(riskColor(2)).toBe(riskColor(1));
    expect(riskColor(0.6)).toMatch(/^#[0-9a-f]{6}$/);
  });

  it('bands and formats probabilities', () => {
    expect(riskBand(0.2)).toBe('Low');
    expect(riskBand(0.5)).toBe('Moderate');
    expect(riskBand(0.9)).toBe('High');
    expect(pct(0.456)).toBe('46%');
  });
});

describe('clinical alerts', () => {
  it('names the rhythm from the pulse rate', () => {
    expect(rhythmOf(45).label).toBe('Sinus bradycardia');
    expect(rhythmOf(72).label).toBe('Normal sinus rhythm');
    expect(rhythmOf(120).label).toBe('Sinus tachycardia');
  });

  it('flags dangerous values and lists critical ones first', () => {
    const flags = clinicalFlags({ ...NEW_CASE, 'EF-TTE': 30, HDL: 20 });
    expect(flags.map((f) => f.key)).toEqual(['EF-TTE', 'HDL']);
    expect(flags[0].level).toBe('danger');
    expect(clinicalFlags(NEW_CASE)).toEqual([]);
  });

  it('warns when an input is outside the training data', () => {
    expect(outOfRange(NEW_CASE)).toEqual([]);
    const ood = outOfRange({ ...NEW_CASE, PR: 40 });
    expect(ood).toHaveLength(1);
    expect(ood[0]).toMatchObject({ key: 'PR', value: 40, range: TRAINING_RANGES.PR });
  });

  it('prefers exact training ranges from the model service when given', () => {
    expect(outOfRange({ ...NEW_CASE, Age: 50 }, { Age: [55, 80] }).map((o) => o.key)).toEqual(['Age']);
  });
});

describe('assessment summary', () => {
  it('reads the CAD result, the flagged vessels and the main contributors', () => {
    const r = mockPredict(typical);
    const lines = assessmentSummary(r);
    expect(lines[0]).toContain('coronary artery disease');
    expect(lines.join(' ')).toContain('LAD');
    expect(lines[lines.length - 1]).toMatch(/^Main contributors: /);
  });

  it("flags each vessel at its own cut-off (screening cut-off for LCX/RCA)", () => {
    const r = mockPredict(typical);
    const set = (t: 'LAD' | 'LCX' | 'RCA', p: number, threshold: number) => {
      r.predictions[t] = { ...r.predictions[t], probability: p, threshold, label: p >= threshold ? 1 : 0 };
    };
    set('LAD', 0.2, 0.5);
    set('LCX', 0.35, 0.3); // above its 30% screening cut-off, below 50%
    set('RCA', 0.25, 0.3);
    const text = assessmentSummary(r).join(' ');
    expect(text).toContain('Stenosis flagged in the LCX (35%)');
    expect(text).not.toContain('RCA (25%)');
  });

  it('lists drivers in order of strength', () => {
    const d = topDrivers(mockPredict(typical), 'CAD', 5);
    for (let i = 1; i < d.length; i++) expect(d[i - 1].shap).toBeGreaterThanOrEqual(d[i].shap);
    expect(d.every((c) => c.shap > 0)).toBe(true);
  });
});
