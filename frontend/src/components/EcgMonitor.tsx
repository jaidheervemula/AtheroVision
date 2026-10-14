import { useEffect, useRef } from 'react';
import type { PatientFeatures } from '../types';
import { contraction, heartbeat, setBpm } from '../beat';
import { rhythmOf } from '../clinical';

// A stylised lead-II strip, not a real recording. Its shape follows the ECG flags entered on
// the form (ST elevation/depression, T inversion, Q waves, LVH, poor R progression, BBB) and
// its rate follows the pulse rate, so the HUD visibly reacts to the patient's inputs.

const g = (x: number, m: number, w: number) => Math.exp(-(((x - m) / w) ** 2));
const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

interface Shape { q: number; r: number; s: number; st: number; t: number; wide: number }

function shapeOf(f: PatientFeatures): Shape {
  const on = (k: string) => Number(f[k]) === 1;
  return {
    q: on('Q Wave') ? 0.32 : 0.08,
    r: (on('Poor R Progression') ? 0.62 : 1) * (on('LVH') ? 1.35 : 1),
    s: 0.26,
    st: on('St Elevation') ? 0.2 : on('St Depression') ? -0.16 : 0,
    t: on('Tinversion') ? -0.24 : 0.28,
    wide: f.BBB && f.BBB !== 'N' ? 1.8 : 1,
  };
}

function sample(phase: number, s: Shape): number {
  const w = s.wide;
  return (
    0.12 * g(phase, 0.12, 0.025) -
    s.q * g(phase, 0.205, 0.008 * w) +
    s.r * g(phase, 0.225, 0.009 * w) -
    s.s * g(phase, 0.248, 0.01 * w) +
    s.st * smoothstep(0.26, 0.29, phase) * (1 - smoothstep(0.36, 0.42, phase)) +
    s.t * g(phase, 0.45, 0.042)
  );
}

export default function EcgMonitor({ features, simulated = false }: { features: PatientFeatures; simulated?: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const shape = useRef<Shape>(shapeOf(features));
  const icon = useRef<HTMLSpanElement>(null);
  // The drawing loop reads the latest waveform shape from this ref.
  useEffect(() => { shape.current = shapeOf(features); }, [features]);
  const bpm = Math.min(160, Math.max(35, Number(features.PR) || 72));
  useEffect(() => setBpm(bpm), [bpm]);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ctx = el.getContext('2d');
    if (!ctx) return;
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const WINDOW = 3.2; // seconds across the strip
    let raf = 0;
    let lastT = performance.now() / 1000;
    let prevPhase = heartbeat.phase;
    let prev: [number, number] | null = null;
    let elapsed = 0;

    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      el.width = el.clientWidth * dpr;
      el.height = el.clientHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, el.clientWidth, el.clientHeight);
      prev = null;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);

    const yOf = (v: number, h: number) => h * 0.62 - v * h * 0.42;

    const frame = () => {
      const now = performance.now() / 1000;
      let dt = Math.min(0.1, now - lastT);
      lastT = now;
      if (reduce) dt = 0;
      const W = el.clientWidth, H = el.clientHeight;
      const steps = Math.max(1, Math.ceil(dt * 240));
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = '#39ff88';
      ctx.shadowColor = '#39ff88';
      ctx.shadowBlur = 8;
      // Follow the shared heartbeat clock, filling in sub-steps between frames.
      const cur = heartbeat.phase;
      const dPhase = (cur - prevPhase + 1) % 1;
      if (icon.current) icon.current.style.transform = `scale(${1 + 0.45 * contraction(cur)})`;
      for (let i = 0; i < steps; i++) {
        const sdt = dt / steps;
        elapsed += sdt;
        const phase = (prevPhase + (dPhase * (i + 1)) / steps) % 1;
        const x = ((elapsed % WINDOW) / WINDOW) * W;
        const y = yOf(sample(phase, shape.current), H);
        if (prev && x < prev[0]) prev = null; // wrapped around
        // erase just ahead of the pen, like a sweep monitor
        ctx.save();
        ctx.shadowBlur = 0;
        ctx.clearRect(x + 1, 0, 14, H);
        ctx.restore();
        if (prev) {
          ctx.beginPath();
          ctx.moveTo(prev[0], prev[1]);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
        prev = [x, y];
      }
      prevPhase = cur;
      if (reduce && !prev) {
        // static strip for reduced motion
        ctx.beginPath();
        for (let x = 0; x <= W; x += 1) {
          const ph = ((x / W) * WINDOW * heartbeat.bpm / 60) % 1;
          const y = yOf(sample(ph, shape.current), H);
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        prev = [W, 0];
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  const flags = [
    Number(features['St Elevation']) === 1 && 'ST↑',
    Number(features['St Depression']) === 1 && 'ST↓',
    Number(features.Tinversion) === 1 && 'T-INV',
    Number(features['Q Wave']) === 1 && 'Q',
    Number(features.LVH) === 1 && 'LVH',
    Number(features['Poor R Progression']) === 1 && 'PRP',
    features.BBB && features.BBB !== 'N' && String(features.BBB),
  ].filter(Boolean) as string[];

  return (
    <div className="ecg">
      <div className="ecg__head">
        <span className="ecg__lead">Lead II</span>
        {simulated && <span className="sim-tag">What-if</span>}
        <span className="ecg__bpm"><span ref={icon} className="ecg__heart">♥</span>{Math.round(bpm)} BPM</span>
        <span className={`ecg__rhythm ecg__rhythm--${rhythmOf(bpm).level}`}>{rhythmOf(bpm).label}</span>
        <span className="ecg__flags">
          {flags.length ? flags.map((f) => <b key={f}>{f}</b>) : <i>No ECG abnormalities entered</i>}
        </span>
        <span className="ecg__note">Schematic trace drawn from the ECG findings entered</span>
      </div>
      <canvas ref={canvas} className="ecg__canvas" aria-label={`Schematic ECG at ${Math.round(bpm)} beats per minute`} />
    </div>
  );
}
