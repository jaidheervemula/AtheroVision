import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { PatientFeatures, PredictionResponse } from '../types';
import { clinicalFlags, outOfRange } from '../clinical';

// App bar + patient banner, modelled on clinical systems: the banner keeps the case identity
// and the key numbers in view at all times, and every control in the bar does something real.

interface Props {
  mode: 'mock' | 'api';
  result: PredictionResponse | null;
  updatedAt: Date | null;
  loading: boolean;
  caseId: string;
  caseLabel: string;
  features: PatientFeatures;
  ranges?: Record<string, [number, number]>;
  simulated: boolean;
  /** Set when the model service can't be reached or rejected the input. */
  error?: string | null;
  onNewCase: () => void;
  onExport: () => void;
  /** Export was requested and is waiting for the explanation to finish. */
  exportPending?: boolean;
  /** Start the guided demo. */
  onPresent?: () => void;
}

function Logo() {
  return (
    // Same mark as the browser icon (public/favicon.svg): an eye whose iris is a coronary artery
    // in cross-section: red wall, amber plaque crescent, narrowed lumen.
    <svg className="logo" viewBox="0 0 64 64" width="34" height="34" aria-hidden>
      <rect x="2" y="2" width="60" height="60" rx="15" fill="#0f1f19" stroke="#2f4b3f" strokeWidth="2" />
      <path d="M5 32C11 22.5 21 17 32 17S53 22.5 59 32" fill="none" stroke="#5fe0a0" strokeWidth="4.4" strokeLinecap="round" />
      <path d="M5 32C11 41.5 21 47 32 47S53 41.5 59 32" fill="none" stroke="#5fe0a0" strokeWidth="4.4" strokeLinecap="round" opacity="0.6" />
      <circle cx="32" cy="32" r="13" fill="#ff5a6e" />
      <circle cx="32" cy="32" r="9.2" fill="#f2c14e" />
      <circle cx="34.6" cy="29.4" r="6.6" fill="#0f1f19" />
    </svg>
  );
}

/** Small popover anchored to a button; closes on outside click or Escape. */
function Popover({ label, button, children, className = '' }: { label: string; button: ReactNode; children: ReactNode; className?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, [open]);
  return (
    <div className={`pop ${className}`} ref={ref}>
      <button className="appbar__btn" aria-expanded={open} aria-haspopup="dialog" aria-label={label} onClick={() => setOpen((o) => !o)}>
        {button}
      </button>
      {open && <div className="pop__panel" role="dialog" aria-label={label}>{children}</div>}
    </div>
  );
}

function useNow(intervalMs = 5000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), intervalMs); return () => clearInterval(t); }, [intervalMs]);
  return now;
}

function ago(d: Date | null, now: number) {
  if (!d) return 'Waiting for first prediction';
  const s = Math.max(0, Math.round((now - d.getTime()) / 1000));
  if (s < 10) return 'Updated just now';
  if (s < 60) return `Updated ${s} s ago`;
  const m = Math.round(s / 60);
  return `Updated ${m} min ago`;
}

export const SHORTCUTS: [string, string][] = [
  ['1', 'Overall CAD'], ['2', 'LAD'], ['3', 'LCX'], ['4', 'RCA'],
  ['R', 'Reset the 3D view'], ['G', 'Glow on or off'], ['E', 'Export report'], ['N', 'New case'], ['P', 'Guided demo (→ next, ← back, Esc exit)'], ['?', 'Show shortcuts'],
];

export default function AppHeader(p: Props) {
  const now = useNow();
  const f = p.features;
  const flags = clinicalFlags(f);
  const danger = flags.filter((x) => x.level === 'danger').length;
  const warn = flags.length - danger;
  const ood = outOfRange(f, p.ranges).length;

  const field = (label: string, value: ReactNode, tone?: 'high' | 'low') => (
    <div className={`banner__field${tone ? ` banner__field--${tone}` : ''}`}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
  const range = (v: number, lo: number, hi: number) => (v > hi ? 'high' : v < lo ? 'low' : undefined);
  const bp = Number(f.BP), pr = Number(f.PR), ef = Number(f['EF-TTE']), bmi = Number(f.BMI);

  return (
    <header className="header">
      <div className="appbar">
        <div className="appbar__brand">
          <Logo />
          <div>
            <h1>AtheroVision</h1>
            <p>Coronary risk assessment</p>
          </div>
        </div>

        <div className="appbar__actions">
          {p.loading && <span className="spinner" aria-label="Updating prediction" />}

          <Popover label="Model status" className="pop--model" button={
            <>
              <span className={`status-dot status-dot--${p.error && p.mode === 'api' ? 'error' : p.mode}`} />
              {p.mode === 'mock' ? 'Demo model' : p.error ? 'Model offline' : 'Live model'}
              <svg className="chev" viewBox="0 0 10 6" width="10" height="6" aria-hidden><path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
            </>
          }>
            <h3>Prediction model</h3>
            <dl className="pop__list">
              <div><dt>Source</dt><dd>{p.mode === 'mock' ? 'Built-in demo model' : 'Model service (API)'}</dd></div>
              <div><dt>Models</dt><dd>{p.result?.model_version ?? '—'}</dd></div>
              <div><dt>Targets</dt><dd>CAD, LAD, LCX, RCA (independent models)</dd></div>
              <div><dt>Explanations</dt><dd>{p.mode === 'mock'
                ? 'Additive, log-odds'
                : 'Exact SHAP values (log-odds) for CAD, LCX, RCA; approximate single-input impact for LAD (random forest)'}</dd></div>
              <div><dt>Flagged from</dt><dd>{p.result
                ? (['CAD', 'LAD', 'LCX', 'RCA'] as const).map((t) => `${t} ${Math.round(p.result!.predictions[t].threshold * 100)}%`).join(', ')
                : '—'}</dd></div>
              <div><dt>Last prediction</dt><dd>{p.updatedAt ? p.updatedAt.toLocaleTimeString() : '—'}</dd></div>
            </dl>
            {p.mode === 'mock' && (
              <p className="pop__note">The demo model uses placeholder weights so the interface can be shown before the trained models are connected. Do not interpret its numbers clinically.</p>
            )}
            {p.mode === 'api' && p.error && <p className="pop__note pop__note--error">{p.error}</p>}
            {p.mode === 'api' && !p.error && p.result?.disclaimer && <p className="pop__note">{p.result.disclaimer}</p>}
          </Popover>

          <Popover label="Keyboard shortcuts" className="pop--keys" button={
            <svg viewBox="0 0 20 14" width="18" height="13" aria-hidden><rect x="0.75" y="0.75" width="18.5" height="12.5" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.5" /><path d="M4 4.5h1M7.5 4.5h1M11 4.5h1M14.5 4.5h1M4 7h1M7.5 7h1M11 7h1M14.5 7h1M6 9.8h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
          }>
            <h3>Keyboard shortcuts</h3>
            <dl className="pop__keys">
              {SHORTCUTS.map(([k, d]) => <div key={k}><dt><kbd>{k}</kbd></dt><dd>{d}</dd></div>)}
            </dl>
          </Popover>

          {p.onPresent && (
            <button className="appbar__btn" onClick={p.onPresent} title="Guided demo for presenting (P)">
              <svg viewBox="0 0 14 14" width="12" height="12" aria-hidden><path d="M3.5 2.2v9.6L11.5 7z" fill="currentColor" /></svg>
              Guided demo
            </button>
          )}
          <button className="appbar__btn" onClick={p.onNewCase}>
            <svg viewBox="0 0 14 14" width="13" height="13" aria-hidden><path d="M7 1.5v11M1.5 7h11" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
            New case
          </button>
          <button className="appbar__btn appbar__btn--primary" onClick={p.onExport} disabled={p.exportPending} aria-busy={p.exportPending}>
            <svg viewBox="0 0 14 14" width="13" height="13" aria-hidden><path d="M7 1.5v7.5M3.8 6l3.2 3.2L10.2 6M2 12.5h10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
            {p.exportPending ? 'Preparing report…' : 'Export report'}
          </button>
        </div>
      </div>

      <div className="banner" aria-label="Current case">
        <div className="banner__case">
          <span className="banner__id">{p.caseId}</span>
          <span className="banner__label">{p.caseLabel}</span>
        </div>
        <dl className="banner__fields">
          {field('Age', `${f.Age} y`)}
          {field('Sex', String(f.Sex))}
          {field('BMI', bmi.toFixed(1), range(bmi, 18.5, 25))}
          {field('Systolic BP', `${bp} mmHg`, range(bp, 90, 130))}
          {field('Pulse', `${pr} bpm`, range(pr, 60, 100))}
          {field('EF', `${ef}%`, range(ef, 55, 70))}
        </dl>
        <div className="banner__right">
          {p.simulated && <span className="tag tag--warn">What-if shown</span>}
          {danger > 0 && <span className="tag tag--danger">{danger} critical</span>}
          {warn > 0 && <span className="tag tag--warn">{warn} {warn === 1 ? 'warning' : 'warnings'}</span>}
          {ood > 0 && <span className="tag tag--muted">{ood} outside training range</span>}
          <span className="banner__time" title={p.updatedAt?.toLocaleString()}>
            <span className={`live-dot${p.loading ? ' live-dot--busy' : ''}`} />
            {ago(p.updatedAt, now)}
          </span>
        </div>
      </div>
    </header>
  );
}
