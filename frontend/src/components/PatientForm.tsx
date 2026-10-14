import { useEffect, useState } from 'react';
import { FEATURES, GROUPS, PRESETS, type FeatureDef } from '../features';
import type { FeatureValue, PatientFeatures } from '../types';

interface Props {
  features: PatientFeatures;
  activePreset: string | null;
  onChange: (key: string, value: FeatureValue) => void;
  onPreset: (id: string) => void;
  /** Fields none of the connected models use (from the model service). */
  unusedInputs?: string[];
  /** Load a real patient from the training dataset (only with the model service). */
  dataset?: { onLoad: (row: number) => void; row: number | null; busy: boolean; error: string | null; nRows: number };
}

function DatasetLoader({ onLoad, row, busy, error, nRows }: NonNullable<Props['dataset']>) {
  // Remounted (via key) when a new row loads, so the box always shows the loaded row.
  const [draft, setDraft] = useState(row == null ? '' : String(row));
  const n = Number(draft);
  const valid = draft.trim() !== '' && Number.isInteger(n) && n >= 0 && n < nRows;
  return (
    <form
      className="dataset"
      aria-label="Load a patient from the training dataset"
      onSubmit={(e) => { e.preventDefault(); if (valid) onLoad(n); }}
    >
      <label htmlFor="dataset-row" className="dataset__label" title="A real patient from the Z-Alizadeh Sani dataset, with the angiography result to compare against">
        Dataset patient
      </label>
      <input
        id="dataset-row" type="number" min={0} max={nRows - 1} step={1} placeholder="Row"
        value={draft} onChange={(e) => setDraft(e.target.value)} aria-invalid={draft !== '' && !valid ? true : undefined}
      />
      <button className="chip" type="submit" disabled={!valid || busy}>{busy ? 'Loading…' : 'Load'}</button>
      <button className="chip" type="button" disabled={busy} onClick={() => onLoad(Math.floor(Math.random() * nRows))}>Random</button>
      {error && <p className="dataset__error" role="alert">{error}</p>}
    </form>
  );
}

const UnusedTag = () => <em className="field__unused" title="None of the four models uses this input">not used by models</em>;

function Field({ def, value, onChange, unused }: { def: FeatureDef; value: FeatureValue; onChange: Props['onChange']; unused: boolean }) {
  const id = `f-${def.key.replace(/\W+/g, '-')}`;
  if (def.kind === 'bool') {
    return (
      <label className="field field--bool" htmlFor={id}>
        <span>{def.label}{unused && <UnusedTag />}</span>
        <input
          id={id}
          type="checkbox"
          className="switch"
          checked={Number(value) === 1}
          disabled={def.derived}
          onChange={(e) => onChange(def.key, e.target.checked ? 1 : 0)}
        />
      </label>
    );
  }
  if (def.kind === 'select') {
    return (
      <label className="field" htmlFor={id}>
        <span>{def.label}{unused && <UnusedTag />}</span>
        <select id={id} value={String(value)} onChange={(e) => onChange(def.key, e.target.value)}>
          {def.options!.map((o) => <option key={o} value={o}>{o === 'N' ? 'None' : o}</option>)}
        </select>
      </label>
    );
  }
  return <NumberField def={def} value={Number(value)} onChange={onChange} unused={unused} />;
}

/**
 * Numeric input that only updates the prediction with a valid, in-range number. While the text is
 * blank or out of range it shows the allowed range, and on leaving the field it reverts to the last
 * valid value, so a half-typed entry never produces a prediction for, say, a 0-year-old.
 */
function NumberField({ def, value, onChange, unused }: { def: FeatureDef; value: number; onChange: Props['onChange']; unused: boolean }) {
  const id = `f-${def.key.replace(/\W+/g, '-')}`;
  const [draft, setDraft] = useState(String(value));
  const [focused, setFocused] = useState(false);
  useEffect(() => { if (!focused) setDraft(String(value)); }, [value, focused]);

  const min = def.min ?? -Infinity;
  const max = def.max ?? Infinity;
  const n = draft.trim() === '' ? NaN : Number(draft);
  const notWhole = !!def.integer && Number.isFinite(n) && !Number.isInteger(n);
  const invalid = !def.derived && (!Number.isFinite(n) || n < min || n > max || notWhole);
  const out = def.normal && (value < def.normal[0] || value > def.normal[1]);
  const hintId = `${id}-hint`;

  return (
    <label className={`field${out ? ' field--out' : ''}${invalid ? ' field--invalid' : ''}`} htmlFor={id}>
      <span>{def.label}{def.derived && <em> (auto)</em>}{unused && <UnusedTag />}</span>
      <span className="field__input">
        <input
          id={id}
          type="number"
          inputMode="decimal"
          min={def.min}
          max={def.max}
          step={def.step}
          value={def.derived ? Math.round(value * 10) / 10 : draft}
          readOnly={def.derived}
          aria-invalid={invalid || undefined}
          aria-describedby={invalid ? hintId : undefined}
          onFocus={() => setFocused(true)}
          onBlur={() => { setFocused(false); setDraft(String(value)); }}
          onChange={(e) => {
            const text = e.target.value;
            setDraft(text);
            const v = Number(text);
            if (text.trim() !== '' && Number.isFinite(v) && v >= min && v <= max && (!def.integer || Number.isInteger(v))) onChange(def.key, v);
          }}
        />
        {def.unit && <span className="field__unit">{def.unit}</span>}
      </span>
      {invalid && (
        <span className="field__hint" id={hintId} role="alert">
          {notWhole ? 'Enter a whole number' : `Enter ${def.min}–${def.max}${def.unit ? ` ${def.unit}` : ''}`}
        </span>
      )}
    </label>
  );
}

export default function PatientForm({ features, activePreset, onChange, onPreset, unusedInputs = [], dataset }: Props) {
  const unused = new Set(unusedInputs);
  return (
    <section className="panel form" aria-label="Patient input">
      <header className="panel__head">
        <h2>Patient</h2>
        <span className="muted">Predictions update as you edit</span>
      </header>
      <div className="presets" role="group" aria-label="Example patients">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            className={`chip${activePreset === p.id ? ' chip--on' : ''}`}
            title={p.note}
            onClick={() => onPreset(p.id)}
          >
            {p.label}
          </button>
        ))}
      </div>
      {dataset && <DatasetLoader key={dataset.row ?? 'none'} {...dataset} />}
      <div className="form__groups">
        {GROUPS.map((g) => (
          <details key={g.id} open={g.open} className="group">
            <summary>{g.label}</summary>
            <div className="group__fields">
              {FEATURES.filter((f) => f.group === g.id).map((def) => (
                <Field key={def.key} def={def} value={features[def.key]} onChange={onChange} unused={unused.has(def.key)} />
              ))}
            </div>
          </details>
        ))}
      </div>
    </section>
  );
}
